import json
from typing import Any, List, Optional

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai_messages_token_helper import build_messages, get_token_limit

from approaches.approach import Approach, ThoughtStep
from core.authentication import AuthenticationHelper
from braintrust import traced

import aiohttp

class RetrieveThenReadApproach(Approach):
    """
    Simple retrieve-then-read implementation, using the AI Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """

    system_chat_template = (
        "You are a helpful, chatty HR help desk assistant helping employees with their HR questions. "
        + "Use 'you' to refer to the individual asking the questions even if they ask with 'I'. "
        + "Answer the following question using only the data provided in the sources below. "
        + "Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. "
        + "If you cannot answer using the sources below, say you don't know. Use below example to answer"
    )

    # shots/sample conversation
    question = """
'What is the deductible for the employee plan for a visit to Overlake in Bellevue?'

Sources:
info1.txt: deductibles depend on whether you are in-network or out-of-network. In-network deductibles are $500 for employee and $1000 for family. Out-of-network deductibles are $1000 for employee and $2000 for family.
info2.pdf: Overlake is in-network for the employee plan.
info3.pdf: Overlake is the name of the area that includes a park and ride near Bellevue.
info4.pdf: In-network institutions include Overlake, Swedish and others in the region.
"""
    answer = "In-network deductibles are $500 for employee and $1000 for family [info1.txt] and Overlake is in-network for the employee plan [info2.pdf][info4.pdf]."

    def __init__(
        self,
        *,
        search_client: SearchClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_model: str,
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_dimensions: int,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
    ):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.chatgpt_token_limit = get_token_limit(chatgpt_model, self.ALLOW_NON_GPT_MODELS)
        self.NO_RESPONSE = "0"

    @traced
    async def run(
        self,
        messages: list[ChatCompletionMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> dict[str, Any]:
        q = messages[-1]["content"]
        if not isinstance(q, str):
            raise ValueError("The most recent message content must be a string.")
        overrides = context.get("overrides", {})
        seed = overrides.get("seed", None)
        auth_claims = context.get("auth_claims", {})
        use_text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = True if overrides.get("semantic_ranker") else False
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top", 3)
        minimum_search_score = overrides.get("minimum_search_score", 0.0)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)
        filter = self.build_filter(overrides, auth_claims)

        # If retrieval mode includes vectors, compute an embedding for the query
        vectors: list[VectorQuery] = []
        if use_vector_search:
            vectors.append(await self.compute_text_embedding(q))

        results = await self.search(
            top,
            q,
            filter,
            vectors,
            use_text_search,
            use_vector_search,
            use_semantic_ranker,
            use_semantic_captions,
            minimum_search_score,
            minimum_reranker_score,
        )

        # Process results
        sources_content = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)

        # Append user message
        content = "\n".join(sources_content)
        user_content = q + "\n" + f"Sources:\n {content}"

        response_token_limit = 1024
        updated_messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=overrides.get("prompt_template", self.system_chat_template),
            few_shots=[{"role": "user", "content": self.question}, {"role": "assistant", "content": self.answer}],
            new_user_content=user_content,
            max_tokens=self.chatgpt_token_limit - response_token_limit,
            fallback_to_default=self.ALLOW_NON_GPT_MODELS,
        )

        chat_completion = await self.openai_client.chat.completions.create(
            # Azure OpenAI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            messages=updated_messages,
            temperature=overrides.get("temperature", 0.3),
            max_tokens=response_token_limit,
            n=1,
            seed=seed,
        )

        data_points = {"text": sources_content}
        extra_info = {
            "data_points": data_points,
            "thoughts": [
                ThoughtStep(
                    "Search using user query",
                    q,
                    {
                        "use_semantic_captions": use_semantic_captions,
                        "use_semantic_ranker": use_semantic_ranker,
                        "top": top,
                        "filter": filter,
                        "use_vector_search": use_vector_search,
                        "use_text_search": use_text_search,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result.serialize_for_results() for result in results],
                ),
                ThoughtStep(
                    "Prompt to generate answer",
                    updated_messages,
                    (
                        {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                        if self.chatgpt_deployment
                        else {"model": self.chatgpt_model}
                    ),
                ),
            ],
        }

        agent_tools: List[ChatCompletionToolParam] = [
            {
                "type": "function", 
                "function": {
                    "name": "get_weather",
                    "description": "Retrieves the weather for a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lat": {
                                "type": "number",
                                "description": "Latitude"
                            },
                            "lng": {
                                "type": "number",
                                "description": "Longitude"
                            },
                            "location": {
                                "type": "string",
                                "description": "Name of the location"
                            }
                        },
                        "required": ["lat", "lng", "location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_joke",
                    "description": "Retrieves and tells a joke",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]

        # Call agent tools if chat_completion starts with "I don't know"
        agent_response = None
        if chat_completion.choices[0].message.content.startswith("I don't "):
            chat_coroutine = await self.openai_client.chat.completions.create(
                # Azure OpenAI takes the deployment name as the model name
                model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
                messages=messages,
                temperature=overrides.get("temperature", 0.3),
                max_tokens=response_token_limit,
                n=1,
                stream=False,
                tools=agent_tools,
                seed=seed,
            )
            response_message = chat_coroutine.choices[0].message

            if response_message.tool_calls:
                for tool in response_message.tool_calls:
                    if tool.type != "function":
                        continue
                    function = tool.function
                    if function.name == "search_sources":
                        arg = json.loads(function.arguments)
                        search_query = arg.get("search_query", self.NO_RESPONSE)
                        if search_query != self.NO_RESPONSE:
                            agent_response = search_query
                    elif function.name == "get_weather":
                        arg = json.loads(function.arguments)
                        lat = arg.get("lat", self.NO_RESPONSE)
                        lng = arg.get("lng", self.NO_RESPONSE)
                        location = arg.get("location", self.NO_RESPONSE)
                        if lat != self.NO_RESPONSE and lng != self.NO_RESPONSE and location != self.NO_RESPONSE:
                            weather_data = await self.get_weather(lat, lng, location)
                            # Extract weather metrics from response
                            temp = weather_data['current']['temperature_2m']
                            wind = weather_data['current']['wind_speed_10m']
                            precip = weather_data['current'].get('precipitation', 0)
                            time = weather_data['current']['time']
                            weather_report = f"Current weather in {location}: {temp}°F with winds at {wind} mph and {precip} inches of precipitation."                            
                        else:
                            weather_report = "Please provide a valid location."
                        agent_response = (f"{agent_response}\n\n{weather_report}" if agent_response is not None else weather_report)
                    elif function.name == "get_joke":
                        joke_data = await self.get_joke()
                        agent_response = (f"{agent_response}\n\nHere's a joke: {joke_data['joke']}" 
                                         if agent_response is not None 
                                         else f"Here's a joke:\n\n{joke_data['joke']}")
        return {
            "message": {
                "content": agent_response if agent_response is not None else chat_completion.choices[0].message.content,
                "role": chat_completion.choices[0].message.role,
            },
            "context": extra_info,
            "session_state": session_state,
        }            

    async def get_weather(self, lat: float, lng: float, location: str) -> dict:
        """Retrieves weather data for given location"""
        async with aiohttp.ClientSession() as session:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current=precipitation&current=temperature_2m,wind_speed_10m&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
            async with session.get(url) as response:
                return await response.json()

    async def get_joke(self) -> dict:
        """Retrieves and tells a random joke"""
        async with aiohttp.ClientSession() as session:
            url = "https://official-joke-api.appspot.com/random_joke"
            async with session.get(url) as response:
                joke_data = await response.json()
                return {
                    "joke": f"{joke_data['setup']} - {joke_data['punchline']}"
                }