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
        "You are an expert Python developer specializing in writing high-quality, "
        "effective unit tests. Your task is to improve an existing test file to "
        "increase its test coverage.\n\n"
        "You will be provided with:\n"
        "1. The source code of a Python file.\n"
        "2. The content of its corresponding test file.\n"
        "3. A list of line numbers that are currently not covered by tests.\n"
        "4. (Optional) The content of other test files from the same project to be "
        "used as a style reference.\n\n"
        "Your goal is to **update the existing test file** by adding new tests or "
        "modifying existing ones to cover the specified `uncovered lines`.\n\n"
        "**Instructions:**\n"
        "- **Analyze the Style:** If other test files are provided, carefully "
        "analyze their structure, mocking techniques (e.g., `unittest.mock`), "
        "assertion styles (e.g., `assert`, `self.assertEqual`), and overall "
        "organization. Apply this same style to the new tests you write.\n"
        "- **Focus on Coverage:** Your primary goal is to write tests that execute "
        "the code on the `uncovered lines`.\n"
        "- **Maintain Existing Tests:** Do not remove or break existing, valid "
        "tests in the file.\n"
        "- **Complete File:** Your response must be the **full, complete content** "
        "of the updated test file.\n"
        "- **Code Only:** Do not include any explanations, comments, or markdown "
        "formatting in your response. It must be only valid Python code."
    )
    user_msg: str = f"""
File to be tested: {file_name}
Uncovered lines to be covered: {coverage_lines}

### Source Code ###
{source_code}

### Current Test File Content ###
{test_code}

### Other Tests for Style Reference ###
{other_tests_content}
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
