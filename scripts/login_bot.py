import os
import time
from typing import Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def create_driver(chromedriver_path: Optional[str] = None) -> webdriver.Chrome:
    """
    Create a Chrome WebDriver with minimal bot-evasion tweaks.
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )

    if not chromedriver_path:
        # Default to repo root chromedriver.exe on Windows if present
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(repo_root, 'chromedriver.exe')
        chromedriver_path = candidate if os.path.exists(candidate) else 'chromedriver'

    service = Service(chromedriver_path)
    return webdriver.Chrome(service=service, options=options)


def fast_login_attempt(driver: webdriver.Chrome, base_url: str, aadhaar_number: str = '123456789123') -> bool:
    """
    Perform a fast, bot-like login attempt to try triggering CAPTCHA.
    Returns True if CAPTCHA overlay is detected, False otherwise.
    """
    driver.get(base_url)

    wait = WebDriverWait(driver, 10)
    aadhaar_input = wait.until(EC.presence_of_element_located((By.ID, 'aadhaarNumber')))

    # Type very quickly (no per-key delay)
    aadhaar_input.clear()
    aadhaar_input.send_keys(aadhaar_number)

    # Immediately submit without moving the mouse to simulate bot-like behavior
    submit_btn = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    submit_btn.click()

    # The UI runs ML and potentially shows a modal. Give it a short time window.
    time.sleep(0.5)

    # Heuristic: look for the CAPTCHA modal content or Verify button
    try:
        wait_short = WebDriverWait(driver, 3)
        wait_short.until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    # Either the modal title text or the Verify button in the modal
                    "//div[contains(., 'Bot Detected!')] | //button[normalize-space()='Verify']",
                )
            )
        )
        return True
    except Exception:
        return False


def main():
    base_url = os.environ.get('LOGIN_URL', 'http://localhost:5173/')
    attempts = int(os.environ.get('ATTEMPTS', '3'))

    driver = create_driver()
    try:
        captcha_seen = False
        for i in range(1, attempts + 1):
            seen = fast_login_attempt(driver, base_url)
            print(f"Attempt {i}: CAPTCHA shown = {seen}")
            captcha_seen = captcha_seen or seen

            # Small pause between attempts to allow logs/ML to process
            time.sleep(1.0)

        if not captcha_seen:
            print("No CAPTCHA detected. Try increasing ATTEMPTS or behaving more bot-like (e.g., run faster).")
        else:
            print("CAPTCHA was triggered in at least one attempt.")
    finally:
        driver.quit()


if __name__ == '__main__':
    main()


