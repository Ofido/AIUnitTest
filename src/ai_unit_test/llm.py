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
    other_tests_content: str,
    test_style: str,
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

    system_msg_base = (
        "You are an expert Python test developer. Your task is to write new unit tests to cover missing lines "
        "in a given code chunk. You must follow the style of existing tests provided as reference. "
        "Your response must be only the new test code, without any explanations, comments, or markdown formatting. "
        "Your response must be only valid Python code."
    )

    if test_style == "unittest_class":
        system_msg_specific = (
            "Generate a new test method to be added inside a `unittest.TestCase` class. "
            "The method name should start with `test_` and it should accept `self` as its first argument."
        )
    elif test_style == "pytest_function":
        system_msg_specific = "Generate a new test function. The function name should start with `test_`."
    else:
        system_msg_specific = "Generate a new test function or method, adapting to the existing test file's style."

    system_msg = f"{system_msg_base} {system_msg_specific}"

    user_msg: str = f"""Here is the information for the test generation:

<file_to_be_tested>
{file_name}
</file_to_be_tested>

<uncovered_lines>
{coverage_lines}
</uncovered_lines>

<source_code_chunk>
{source_code}
</source_code_chunk>

<existing_tests>
{test_code}
</existing_tests>

<style_reference_tests>
{other_tests_content}
</style_reference_tests>
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
