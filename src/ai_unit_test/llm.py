import logging
import os

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"


async def update_test_with_llm(
    source_code: str,
    test_code: str,
    file_name: str,
    coverage_lines: list[int],
) -> str:
    """Calls the chat model to generate the new test file."""
    logger.info(f"Updating test for {file_name} with LLM.")
    logger.debug(
        f"Source code length: {len(source_code)}, Test code length: {len(test_code)}, Uncovered lines: {coverage_lines}"
    )

    api_key_val: str | None = os.environ.get("OPENAI_API_KEY")
    if not api_key_val:
        logger.error("OPENAI_API_KEY environment variable not set.")
        raise RuntimeError("Set the OPENAI_API_KEY environment variable with your OpenAI key.")

    api_url: str | None = os.environ.get("OPENAI_API_URL")

    client: AsyncOpenAI = AsyncOpenAI(api_key=api_key_val, base_url=api_url)

    system_msg: str = (
        "You are a Python assistant specializing in unit tests. "
        "You will receive the source code, the current test, and the uncovered lines. "
        "Respond only with the complete new content of the test file, in valid Python."
    )
    user_msg: str = f"""
File: {file_name}
Uncovered lines: {coverage_lines}

### Source Code ###
{source_code}

### Current Tests ###
{test_code}
"""
    logger.debug(f"User message for LLM: {user_msg}")
    try:
        rsp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
        )
        response_content: str | None = rsp.choices[0].message.content
        if response_content is None:
            response_content = ""
        logger.debug(f"LLM response received: {response_content[:100]}...")
        return response_content
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        raise
