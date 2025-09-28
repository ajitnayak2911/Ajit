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
from exa_py import Exa
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

@tool
def solve_recaptcha(sitekey: str, page_url: str) -> str:
    """
    Solves Google reCAPTCHA v2 using 2Captcha.
    Arguments:
        sitekey: The sitekey value from the reCAPTCHA iframe
        page_url: The URL of the page where the CAPTCHA is found
    Returns:
        The CAPTCHA solution token to inject into g-recaptcha-response
    """
    captcha_api_key = os.getenv("CAPTCHA_API_KEY")
    print("CAPTCHA API Key:", captcha_api_key)
    if not captcha_api_key:
        raise ValueError("CAPTCHA_API_KEY not found in environment variables")

    response = requests.get("http://2captcha.com/in.php", params={
        "key": captcha_api_key,
        "method": "userrecaptcha",
        "googlekey": sitekey,
        "pageurl": page_url,
        "json": 1
    })
    if response.json().get("status") != 1:
        raise Exception("Failed to submit captcha request to 2Captcha")

    captcha_id = response.json().get("request")
    for _ in range(20):
        time.sleep(5)
        result = requests.get("http://2captcha.com/res.php", params={
            "key": captcha_api_key,
            "action": "get",
            "id": captcha_id,
            "json": 1
        })
        if result.json().get("status") == 1:
            return result.json().get("request")
    raise Exception("Captcha solving timed out")

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
        "- If reCAPTCHA appears, do NOT click the checkbox or verify button directly.\n"
        "  Instead:\n"
        "  - Detect the reCAPTCHA iframe and extract the sitekey from its src attribute.\n"
        "  - Send the sitekey and page URL to a CAPTCHA solving service like 2Captcha.\n"
        "  - Poll until the token is ready.\n"
        "  - Inject the token into the g-recaptcha-response field in the DOM.\n"
        "  - Ensure the CAPTCHA is solved before submitting the form.\n"
        "- Perform human-like typing with varied delays and realistic mouse movement.\n"
        "- Do not reset progress after an interruption. Resume from last completed step.\n"
        "Steps to Perform:\n"
        "1. Navigate to the webpage using Basic Authentication: https://broadridgedigital:broadridge1@www-dev.broadridge.com/testing/br-team/mani/google-recaptcha\n"
        "2. Wait for the full page to load completely.\n"
        "3. Close the cookie banner at the bottom by clicking the \"Close\" button. Wait for the banner to be dismissed.\n"
        "4. Locate the form and ensure all fields are visible and ready.\n"
        f"5. Fill all mandatory form fields with fresh, randomized, and valid data:\n"
        f"   - First Name: {test_data['first_name']}\n"
        f"   - Last Name: {test_data['last_name']}\n"
        f"   - Email: {test_data['email']}\n"
        f"   - Phone Number: {test_data['phone']}\n"
        "   - Country: Open the dropdown and try selecting 'United States' by visible text.\n"
        "     If not successful, use keyboard navigation (ArrowDown + Enter) to select.\n"
        f"   - Comment: {test_data['comment']}\n"
        "6. Detect reCAPTCHA by checking for an iframe with 'recaptcha' in the src.\n"
        "7. DO NOT click the reCAPTCHA checkbox or 'Verify' button directly—this could reveal bot-like behavior.\n"
        "   Instead, follow these steps:\n"
        "   - Extract the sitekey from the iframe's src attribute.\n"
        "   - Use the 2Captcha service to solve the challenge:\n"
        "     - Send your API key, sitekey, and current page URL to https://2captcha.com/in.php.\n"
        "     - Poll https://2captcha.com/res.php until the token is ready (status: 1).\n"
        "   - When received, inject the solution token into the hidden textarea named 'g-recaptcha-response'.\n"
        "   - Also set grecaptcha.getResponse() programmatically if needed.\n"
        "   - Confirm the CAPTCHA is solved before continuing to submit the form.\n"
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
    print(json.dumps(test_result.dict(), indent=2))

    with open("submission_result.json", "w") as f:
        json.dump(test_result.dict(), f, indent=2)

if __name__ == "__main__":
    asyncio.run(SiteValidation())
