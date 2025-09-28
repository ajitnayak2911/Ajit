import asyncio
import os
import random
import json
import time
from faker import Faker
from dotenv import load_dotenv
from browser_use.agent.service import Agent
from browser_use.controller.service import Controller
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import SecretStr, BaseModel
from playwright.async_api import BrowserContext, async_playwright
import requests
from fake_useragent import UserAgent
from langchain.tools import tool, Tool

fake = Faker()

class Formsubmissionresult(BaseModel):
    url_entry_status: str
    cookie_banner_close_status: str
    Talk_to_us_click_status: str
    form_modal_open_status: str
    form_fill_and_submit_status: str
    thank_you_screen_confirmation_message_status: str

controller = Controller(output_model=Formsubmissionresult)

load_dotenv()

def generate_random_test_data():
    return {
        "first_name": fake.first_name(),
        "last_name": fake.last_name(),
        "email": fake.email(),
        "phone": fake.phone_number(),
        "country": "United States",
        "comment": fake.text(max_nb_chars=150)
    }

def get_random_user_agent():
    ua = UserAgent()
    return ua.random

def get_random_proxy():
    return None

async def SiteValidation():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not found in environment variables")

    test_data = generate_random_test_data()

    task = (
            f"Role: You are a Human-Like UI Automation Tester.\n"
            "Your task is to fill out and submit the form on the webpage accurately.\n"
            "Important Instructions:\n"
            "- Absolutely avoid using any numeric indexes (like index: 15). These are unstable and lead to errors.\n"
            "- Always identify elements by their:\n"
            "  - Visible label (e.g., 'First Name')\n"
            "  - Placeholder attribute (e.g., placeholder=\"Enter your name\")\n"
            "  - Name or aria-label (e.g., name=\"email\", aria-label=\"Phone Number\")\n"
            "- Use XPath or CSS selectors only when tied to semantic attributes, not DOM order.\n"
            "- For dropdowns, open the dropdown and select options by their visible text.\n"
            "  If that fails, fall back to simulating key presses to select 'United States'.\n"
            "- Before typing, wait until the element is visible and interactive.\n"
            "- After typing, verify the input value using DOM read-back.\n"
            "- Detect reCAPTCHA by checking for an iframe with 'recaptcha' in the src.\n"
            "- Do NOT attempt to solve or click the CAPTCHA checkbox programmatically to avoid bot detection.\n"
            "- Perform human-like typing with varied delays and realistic mouse movement.\n"
            "- Do not reset progress after an interruption. Resume from last completed step.\n"
            "Steps to Perform:\n"
            #"1. Navigate to the webpage using Basic Authentication: https://broadridgedigital:broadridge1@www-dev.broadridge.com/artificial-intelligence\n"
            "1. Navigate to the webpage https://www.broadridge.com/\n"
            "2. Wait for the full page to load completely.\n"
            "3. Close the cookie banner at the bottom by clicking the \"Close\" button. Wait for the banner to be dismissed.\n"
            "4. Locate the form and ensure all fields are visible and ready.\n"
            f"5. Fill all mandatory form fields with fresh, randomized, and valid data:\n"
            f"   - First Name: {test_data['first_name']}\n"
            f"   - Last Name: TESTTEST\n"
            f"   - Email: {test_data['email']}\n"
            f"   - Phone Number: {test_data['phone']}\n"
            "   - Country: Open the dropdown and try selecting 'United States' by visible text.\n"
            "     If not successful, use keyboard navigation (ArrowDown + Enter) to select.\n"
            f"   - Comment: {test_data['comment']}\n"
            "6. Detect reCAPTCHA by checking for an iframe with 'recaptcha' in the src.\n"
            "7. Do NOT click or solve the reCAPTCHA checkbox or 'Verify' button directly.\n"
            "8. Click the \"Contact us\" button to submit the form.\n"
            "9. After submission, wait for the \"Thank You\" message to appear.\n"
            "10. Capture a screenshot of the confirmation page.\n"
            "11. Close the browser session.\n"
            "Execution Rules:\n"
            "- Wait until elements are visible and interactive before acting.\n"
            "- Skip any optional or hidden fields that might block submission.\n"
            "- Do not reuse data across runs.\n"
            "- Follow all steps in order.\n"
    )

    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))
    agent = Agent(
        task=task,
        llm=llm,
        controller=controller,
        use_vision=True
    )

    print("Starting agent task...")
    history = await agent.run()
    test_result = history.final_result()

    print("Test result:")
    if isinstance(test_result, str):
        test_result = json.loads(test_result)

        print(json.dumps(test_result, indent=2))

    with open("submission_result.json", "w") as f:
        json.dump(test_result, f, indent=2)


if __name__ == "__main__":
    asyncio.run(SiteValidation())
