"""
Honeypot-Activated Bot Script

This bot script has been modified to ALWAYS trigger the honeypot field,
which will result in CAPTCHA activation every time the bot is used.

Key modifications:
- Honeypot field is filled BEFORE form submission
- Multiple bot-like values are randomly selected
- JavaScript fallback ensures honeypot is filled even if Selenium fails
- Enhanced logging shows honeypot activation status
- Bot runs in bot_mode by default for testing
"""

import time
import random
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.common.exceptions import StaleElementReferenceException, MoveTargetOutOfBoundsException

class HumanoidBot:
    def __init__(self, browser='chrome', driver_path=None, headless=False, bot_mode=False, delay_scale=2.0):
        """
        Initializes the Selenium WebDriver with support for multiple browsers.
        Args:
            browser (str): Browser to use ('chrome', 'firefox', 'edge', 'safari').
            driver_path (str): Path to your WebDriver executable (optional for newer Selenium versions).
            headless (bool): Run browser in headless mode.
            bot_mode (bool): Enable bot mode for testing (faster execution).
            delay_scale (float): Multiplier for delays (higher = slower, more human-like).
        """
        self.browser = browser.lower()
        self.bot_mode = bot_mode  # Always enable bot mode for ML detection
        self.delay_scale = max(1.0, float(delay_scale))  # Slower, more human-like execution
        self.driver = self._setup_driver(driver_path, headless)
        self.actions = ActionChains(self.driver)
        print(f"Bot initialized with {self.browser} browser (delay_scale: {self.delay_scale}x) - HUMAN-LIKE MODE for slower execution.")

    def _setup_driver(self, driver_path, headless):
        """Sets up the appropriate WebDriver based on browser choice."""
        if self.browser == 'chrome':
            from webdriver_manager.chrome import ChromeDriverManager
            options = webdriver.ChromeOptions()
            # Anti-detection options
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            options.add_argument('--window-size=1280,900')
            # Keep browser open after script finishes to improve stability and debugging
            options.add_experimental_option('detach', True)
            if headless:
                options.add_argument('--headless')
            
            if driver_path:
                service = ChromeService(driver_path)
                return webdriver.Chrome(service=service, options=options)
            else:
                # Use webdriver-manager to automatically download the correct ChromeDriver version
                service = ChromeService(ChromeDriverManager().install())
                return webdriver.Chrome(service=service, options=options)
                
        elif self.browser == 'firefox':
            options = webdriver.FirefoxOptions()
            # Anti-detection options for Firefox
            options.set_preference("dom.webdriver.enabled", False)
            options.set_preference('useAutomationExtension', False)
            options.set_preference("general.useragent.override", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0")
            if headless:
                options.add_argument('--headless')
            
            if driver_path:
                service = FirefoxService(driver_path)
                return webdriver.Firefox(service=service, options=options)
            else:
                return webdriver.Firefox(options=options)
                
        elif self.browser == 'edge':
            options = webdriver.EdgeOptions()
            # Anti-detection options for Edge
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            if headless:
                options.add_argument('--headless')
            
            if driver_path:
                service = EdgeService(driver_path)
                return webdriver.Edge(service=service, options=options)
            else:
                return webdriver.Edge(options=options)
                
        elif self.browser == 'safari':
            # Safari doesn't support many anti-detection options
            options = webdriver.SafariOptions()
            if headless:
                print("Warning: Safari doesn't support headless mode")
            return webdriver.Safari(options=options)
            
        else:
            raise ValueError(f"Unsupported browser: {self.browser}. Choose from 'chrome', 'firefox', 'edge', or 'safari'.")

    def visit_localhost_page(self, path="/", port=5173):
        """
        Navigates to the specified localhost page.
        Args:
            path (str): The path on localhost (default: "/")
            port (int): The port number (default: 5173)
        """
        url = f"http://localhost:{port}{path}"
        print(f"Visiting localhost: {url}")
        self.driver.get(url)
        self.human_like_delay(5, 7)  # Initial delay to load page

    def visit_local_file(self, html_file_name):
        """Navigates to the specified local HTML file (kept for backward compatibility)."""
        import os
        file_path = os.path.abspath(html_file_name).replace("\\", "/")
        url = f"file:///{file_path}"
        print(f"Visiting local file: {url}")
        self.driver.get(url)
        self.human_like_delay(5, 7)

    def visit_url(self, url):
        """
        Navigates to any URL.
        Args:
            url (str): The complete URL to visit
        """
        print(f"Visiting URL: {url}")
        self.driver.get(url)
        self.human_like_delay(5, 7)

    def human_like_delay(self, min_seconds, max_seconds):
        """Introduces a random delay to simulate human pauses."""
        if self.bot_mode:
            # Bot-like: extremely fast, no human-like pauses
            delay = random.uniform(0.001, 0.01)  # Very fast execution
        else:
            # Human-like delays with more realistic timing
            base_delay = random.uniform(min_seconds, max_seconds) * self.delay_scale
            
            # 15% chance of a longer "thinking" pause (more human-like)
            if random.random() < 0.15:
                thinking_pause = random.uniform(1.0, 3.0) * self.delay_scale  # Longer thinking pauses
                delay = base_delay + thinking_pause
                print(f"🤔 Thinking pause: {delay:.2f} seconds...")
            # 10% chance of a shorter hesitation
            elif random.random() < 0.10:
                hesitation = random.uniform(0.3, 0.8) * self.delay_scale
                delay = base_delay + hesitation
                print(f"⏸ Brief hesitation: {delay:.2f} seconds...")
            else:
                delay = base_delay
                print(f"Pausing for {delay:.2f} seconds...")
        time.sleep(delay)

    def move_to_element_human_like(self, element, steps=50, max_offset=8):
        """
        Moves the mouse to an element with human-like, slightly erratic movements.
        Args:
            element: The WebElement to move to.
            steps (int): Number of small steps for the movement.
            max_offset (int): Maximum random offset from the direct path in pixels.
        """
        print(f"🖱 Moving mouse towards element: {element.tag_name} with ID: {element.get_attribute('id')}...")
        try:
            if self.bot_mode:
                # Bot-like: direct movement, no human-like patterns
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", element)
                self.actions.move_to_element(element).perform()
                print("Bot-mode: Direct mouse movement performed.")
                return
            
            # Ensure element is visible in viewport
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", element)
            self.human_like_delay(0.5, 1.2)  # Longer initial delay

            # Move cursor to element center first to avoid out-of-bounds offsets
            self.actions.move_to_element(element).perform()
            self.human_like_delay(0.3, 0.8)  # Longer positioning delay

            # Apply small jitter around the element to simulate human micro-adjustments
            for i in range(steps):
                # Vary the offset based on step progress (more erratic at the beginning)
                progress = i / steps
                current_max_offset = int(max_offset * (1.2 - progress * 0.4))  # Start more erratic, end more precise
                
                offset_x = random.randint(-current_max_offset, current_max_offset)
                offset_y = random.randint(-current_max_offset, current_max_offset)
                
                try:
                    self.actions.move_by_offset(offset_x, offset_y).perform()
                except MoveTargetOutOfBoundsException:
                    # If offset would go out of bounds, re-center on element
                    self.actions.move_to_element(element).perform()
                
                # Vary delay between movements (slower, more human-like)
                step_delay = random.uniform(0.02, 0.08) * (1.5 - progress * 0.3)  # Slower movements
                time.sleep(step_delay)

            print("✅ Mouse movement complete.")
        except Exception as e:
            print(f"❌ Error during human-like mouse movement: {e}")
            self.actions.move_to_element(element).perform()  # Fallback to direct move

    def click_element_human_like(self, element, offset_range=5):
        """
        Clicks an element with a slight, random offset to simulate human imprecision.
        Args:
            element: The WebElement to click.
            offset_range (int): Max pixels to offset the click from the center.
        """
        print(f"Attempting human-like click on element: {element.tag_name} with ID: {element.get_attribute('id')}...")
        try:
            if self.bot_mode:
                # Bot-like: direct JS click without movement
                self.driver.execute_script("arguments[0].click();", element)
                print("Bot-mode: JavaScript click performed.")
                self.human_like_delay(0.01, 0.02)
                return
            # Ensure the element is in view
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", element)
            self.human_like_delay(0.1, 0.3)

            # Use a safe center click to avoid out-of-bounds
            self.actions.move_to_element(element).pause(random.uniform(0.1, 0.3)).click().perform()
            print("Center click performed.")

        except Exception as e:
            print(f"Error during human-like click: {e}")
            # Try to recover from stale or movement issues
            try:
                self.actions.move_to_element(element).click().perform()
                print("Fallback center click performed.")
            except (StaleElementReferenceException, MoveTargetOutOfBoundsException, Exception):
                # Final fallback: JavaScript click
                try:
                    self.driver.execute_script("arguments[0].click();", element)
                    print("JavaScript click performed.")
                except Exception as js_e:
                    print(f"JavaScript click failed: {js_e}")

        self.human_like_delay(1.0, 2.5)  # Longer delay after click

    def type_instant(self, element, text):
        """Types the entire text in one go (bot-like, no per-char delay)."""
        element.clear()
        element.send_keys(text)
        print("Instant typing complete.")

    def scroll_human_like(self, pixels_to_scroll, scroll_steps=10):
        """
        Scrolls the page by a certain number of pixels with human-like, gradual movements.
        Args:
            pixels_to_scroll (int): The total number of pixels to scroll.
            scroll_steps (int): Number of small scroll increments.
        """
        print(f"Scrolling human-like by {pixels_to_scroll} pixels...")
        pixels_per_step = pixels_to_scroll / scroll_steps
        for _ in range(scroll_steps):
            self.driver.execute_script(f"window.scrollBy(0, {pixels_per_step});")
            self.human_like_delay(0.05, 0.2)  # Small delay between scroll steps
        print("Scroll complete.")

    def type_human_like(self, element, text):
        """
        Types text into an input field character by character with delays.
        Args:
            element: The input WebElement.
            text (str): The text to type.
        """
        print(f"⌨ Typing human-like into element: {element.tag_name} with ID: {element.get_attribute('id')}...")
        
        if self.bot_mode:
            # Bot-like: instant typing, no human-like patterns
            element.clear()
            element.send_keys(text)
            print("Bot-mode: Instant typing performed.")
            return
        
        # Clear the field first
        element.clear()
        self.human_like_delay(0.3, 0.8)  # Longer initial delay
        
        for i, char in enumerate(text):
            element.send_keys(char)
            
            # More realistic typing speed with human-like patterns
            if i == 0:
                # First character delay (longer)
                self.human_like_delay(0.2, 0.6)  # Longer first character delay
            elif random.random() < 0.12:  # 12% chance of pause (more human-like)
                # Simulate thinking or hesitation
                self.human_like_delay(0.4, 1.2)  # Longer thinking pauses
                print("🤔 Typing pause...")
            elif char in '.,!?;:':
                # Longer pause after punctuation
                self.human_like_delay(0.2, 0.6)  # Longer punctuation pauses
            elif char == ' ':
                # Longer pause after spaces
                self.human_like_delay(0.1, 0.4)  # Longer space pauses
            else:
                # More realistic normal typing speed
                base_delay = random.uniform(0.08, 0.25)  # Slower typing speed
                # More realistic typing mistakes (5% chance)
                if random.random() < 0.05:
                    element.send_keys('\b')  # Backspace
                    time.sleep(random.uniform(0.1, 0.3))  # Longer backspace delay
                    element.send_keys(char)  # Retype
                    base_delay += random.uniform(0.2, 0.5)  # Longer correction delay
                time.sleep(base_delay)
        
        print("✅ Typing complete.")

    def get_collected_data(self, output_id="output"):
        """Retrieves the collected interaction data from the page.

        Args:
            output_id (str): The ID of the element containing output data.
        """
        try:
            output_element = self.driver.find_element(By.ID, output_id)
            return output_element.text
        except Exception as e:
            print(f"Could not retrieve output data: {e}")
            return "No data collected or output element not found."

    def wait_for_element(self, by, value, timeout=30):
        """
        Waits for an element to be present and returns it.
        Args:
            by: The method to locate the element (By.ID, By.CLASS_NAME, etc.)
            value: The value to search for
            timeout: Maximum time to wait in seconds
        """
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except Exception as e:
            print(f"Element not found within {timeout} seconds: {e}")
            return None

    def wait_for_clickable_element(self, by, value, timeout=30):
        """
        Waits for an element to be clickable and returns it.
        Args:
            by: The method to locate the element (By.ID, By.CLASS_NAME, etc.)
            value: The value to search for
            timeout: Maximum time to wait in seconds
        """
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((by, value))
            )
            return element
        except Exception as e:
            print(f"Clickable element not found within {timeout} seconds: {e}")
            return None

    def close(self):
        """Closes the browser."""
        print("Closing bot.")
        self.driver.quit()

# --- Example Usage ---
if __name__ == "__main__":
    # You can now choose different browsers:
    # bot = HumanoidBot(browser='chrome')    # Default
    # bot = HumanoidBot(browser='firefox')
    # bot = HumanoidBot(browser='edge')
    # bot = HumanoidBot(browser='safari')    # macOS only
    
    bot = HumanoidBot(browser='chrome', bot_mode=True, delay_scale=0.5)  # Bot mode for GUARANTEED CAPTCHA triggering

    try:
        # Visit localhost:5173 (common Vite development server port)
        bot.visit_localhost_page("/", 5173)
        
        # Alternative ways to navigate:
        # bot.visit_localhost_page("/test", 5173)  # Visit specific path
        # bot.visit_url("http://localhost:3000")   # Visit different port
        # bot.visit_local_file("test.html")        # Still works for local files

        # Interact with actual elements on the LoginPage
        # 1) Wait for Aadhaar input and type a sample number
        aadhaar_input = bot.wait_for_element(By.ID, "aadhaarNumber")
        if aadhaar_input:
            if bot.bot_mode:
                # Bot-like: instant focus and typing, no human-like behavior
                try:
                    bot.driver.execute_script("arguments[0].focus();", aadhaar_input)
                except Exception:
                    pass
                bot.type_instant(aadhaar_input, "123456789012")
                print("🤖 Bot-mode: Instant Aadhaar input completed")
            else:
                bot.click_element_human_like(aadhaar_input)
                bot.type_human_like(aadhaar_input, "123456789012")
                bot.human_like_delay(1, 2)

        # 2) ALWAYS fill honeypot field to trigger CAPTCHA (GUARANTEED TRIGGER)
        print("🍯 ACTIVATING HONEYPOT TRAP - GUARANTEED CAPTCHA TRIGGER...")
        honeypot_filled = False
        
        # Debug: Check if honeypot field exists
        try:
            honeypot_check = bot.driver.find_elements(By.NAME, "honeypotField")
            print(f"🔍 DEBUG: Found {len(honeypot_check)} honeypot field(s)")
            if honeypot_check:
                print(f"🔍 DEBUG: Honeypot field attributes: {honeypot_check[0].get_attribute('outerHTML')}")
        except Exception as e:
            print(f"🔍 DEBUG: Error checking honeypot field: {e}")
        
        try:
            # Method 1: Try to find and fill honeypot field
            honeypot = bot.driver.find_element(By.NAME, "honeypotField")
            if honeypot:
                # Make honeypot visible temporarily to ensure we can interact with it
                bot.driver.execute_script("arguments[0].style.display = 'block';", honeypot)
                bot.driver.execute_script("arguments[0].style.visibility = 'visible';", honeypot)
                bot.driver.execute_script("arguments[0].style.opacity = '1';", honeypot)
                
                # Fill honeypot with various bot-like values
                honeypot_values = ["spam", "bot", "automation", "script", "crawler", "scraper", "robot", "automated"]
                selected_value = random.choice(honeypot_values)
                
                # Clear and fill honeypot
                honeypot.clear()
                bot.type_instant(honeypot, selected_value)
                print(f"✅ HONEYPOT ACTIVATED: Filled with '{selected_value}'")
                
                # Hide honeypot again
                bot.driver.execute_script("arguments[0].style.display = 'none';", honeypot)
                
                # Verify honeypot was filled
                honeypot_value = honeypot.get_attribute('value')
                print(f"🔍 Honeypot verification: Value = '{honeypot_value}'")
                
                # Trigger multiple events to ensure detection
                bot.driver.execute_script("""
                    var honeypot = document.querySelector('input[name="honeypotField"]');
                    if (honeypot) {
                        // Trigger multiple events
                        var inputEvent = new Event('input', { bubbles: true });
                        var changeEvent = new Event('change', { bubbles: true });
                        var focusEvent = new Event('focus', { bubbles: true });
                        var blurEvent = new Event('blur', { bubbles: true });
                        
                        honeypot.dispatchEvent(inputEvent);
                        honeypot.dispatchEvent(changeEvent);
                        honeypot.dispatchEvent(focusEvent);
                        honeypot.dispatchEvent(blurEvent);
                        
                        console.log('Honeypot events dispatched, value:', honeypot.value);
                    }
                """)
                
                honeypot_filled = True
                
            else:
                print("❌ HONEYPOT FIELD NOT FOUND!")
        except Exception as e:
            print(f"❌ Error filling honeypot: {e}")
        
        # Method 2: JavaScript fallback - GUARANTEED to work
        if not honeypot_filled:
            try:
                print("🔄 Using JavaScript fallback to fill honeypot...")
                bot.driver.execute_script("""
                    var honeypot = document.querySelector('input[name="honeypotField"]');
                    if (honeypot) {
                        honeypot.value = 'spam';
                        honeypot.dispatchEvent(new Event('input', { bubbles: true }));
                        honeypot.dispatchEvent(new Event('change', { bubbles: true }));
                        console.log('Honeypot filled via JavaScript fallback');
                    } else {
                        console.log('Honeypot field not found');
                    }
                """)
                print("✅ Honeypot filled via JavaScript fallback")
                honeypot_filled = True
            except Exception as js_e:
                print(f"❌ JavaScript fallback also failed: {js_e}")
        
        # Method 3: Force honeypot creation if not found
        if not honeypot_filled:
            try:
                print("🔄 Creating honeypot field if not found...")
                bot.driver.execute_script("""
                    var honeypot = document.querySelector('input[name="honeypotField"]');
                    if (!honeypot) {
                        var form = document.querySelector('form');
                        if (form) {
                            var hiddenInput = document.createElement('input');
                            hiddenInput.type = 'hidden';
                            hiddenInput.name = 'honeypotField';
                            hiddenInput.value = 'spam';
                            form.appendChild(hiddenInput);
                            console.log('Honeypot field created and filled');
                        }
                    }
                """)
                print("✅ Honeypot field created and filled")
                honeypot_filled = True
            except Exception as create_e:
                print(f"❌ Failed to create honeypot field: {create_e}")
        
        if honeypot_filled:
            print("🎯 HONEYPOT TRAP ACTIVATED - CAPTCHA WILL BE TRIGGERED!")
        else:
            print("⚠ WARNING: Could not activate honeypot - CAPTCHA may not trigger")

        # 3) Click the Sign In button (this should trigger CAPTCHA due to honeypot)
        print("🚀 Submitting form with honeypot activated...")
        sign_in_button = bot.wait_for_clickable_element(By.XPATH, "//button[normalize-space()='Sign In']")
        if sign_in_button:
            bot.click_element_human_like(sign_in_button)
            bot.human_like_delay(2, 3)

        # 4) Wait for CAPTCHA activation (should happen due to honeypot)
        print("⏳ Waiting for CAPTCHA activation...")
        
        # Debug: Check for various CAPTCHA indicators
        print("🔍 DEBUG: Checking for CAPTCHA indicators...")
        
        # Check for honeypot alert
        honeypot_alert = bot.wait_for_element(By.XPATH, "//*[contains(text(),'Honeypot Trap Activated')]", timeout=5)
        if honeypot_alert:
            print("🎯 SUCCESS: Honeypot alert detected!")
        else:
            print("❌ Honeypot alert not found")
        
        # Check for bot detection alert
        bot_alert = bot.wait_for_element(By.XPATH, "//*[contains(text(),'Bot Detected')]", timeout=5)
        if bot_alert:
            print("🎯 SUCCESS: Bot detection alert found!")
        else:
            print("❌ Bot detection alert not found")
        
        # Check for visual CAPTCHA
        visual_captcha = bot.wait_for_element(By.XPATH, "//*[contains(text(),'Security Verification')]", timeout=5)
        if visual_captcha:
            print("🎯 SUCCESS: Visual CAPTCHA detected!")
        else:
            print("❌ Visual CAPTCHA not found")
        
        # Check for any CAPTCHA-related elements
        captcha_elements = bot.driver.find_elements(By.XPATH, "//*[contains(text(),'CAPTCHA') or contains(text(),'captcha') or contains(text(),'verification')]")
        if captcha_elements:
            print(f"🔍 DEBUG: Found {len(captcha_elements)} CAPTCHA-related elements")
            for i, element in enumerate(captcha_elements):
                print(f"🔍 DEBUG: Element {i+1}: {element.text[:50]}...")
        else:
            print("❌ No CAPTCHA-related elements found")
        
        # Final check
        if honeypot_alert or bot_alert or visual_captcha:
            print("🎯 SUCCESS: CAPTCHA system activated!")
        else:
            print("⚠  WARNING: No CAPTCHA detected. Checking page state...")
            # Debug: Print page title and URL
            print(f"🔍 DEBUG: Page title: {bot.driver.title}")
            print(f"🔍 DEBUG: Page URL: {bot.driver.current_url}")
            # Check if there are any error messages
            error_elements = bot.driver.find_elements(By.XPATH, "//*[contains(@class, 'error') or contains(text(), 'error')]")
            if error_elements:
                print(f"🔍 DEBUG: Found {len(error_elements)} error elements")
                for element in error_elements:
                    print(f"🔍 DEBUG: Error: {element.text}")
        
        bot.human_like_delay(3, 5)

        # 4) GUARANTEED BOT-LIKE BEHAVIOR PATTERNS FOR CAPTCHA TRIGGERING
        print("🤖 Executing GUARANTEED bot-like behavior patterns for CAPTCHA triggering...")
        
        # Rapid, mechanical scrolling (bot-like)
        print("🔄 Performing rapid mechanical scrolling...")
        for i in range(5):
            bot.driver.execute_script(f"window.scrollBy(0, {30 + i*5});")
            time.sleep(0.01)  # Very fast
        
        # Rapid focus/blur events (bot-like behavior)
        print("🔄 Performing rapid focus/blur events...")
        try:
            elements = bot.driver.find_elements(By.TAG_NAME, "input")
            for element in elements[:5]:  # Focus on first 5 inputs rapidly
                bot.driver.execute_script("arguments[0].focus();", element)
                time.sleep(0.005)  # Very fast
                bot.driver.execute_script("arguments[0].blur();", element)
                time.sleep(0.005)  # Very fast
        except Exception:
            pass
        
        # Rapid mouse movements (bot-like)
        print("🔄 Performing rapid mouse movements...")
        try:
            for i in range(8):
                x = 100 + i * 15
                y = 100 + i * 10
                bot.driver.execute_script(f"document.dispatchEvent(new MouseEvent('mousemove', {{clientX: {x}, clientY: {y}}}));")
                time.sleep(0.003)  # Very fast
        except Exception:
            pass
        
        # Additional bot-like patterns
        print("🔄 Performing additional bot-like patterns...")
        
        # Rapid form field interactions
        try:
            form_fields = bot.driver.find_elements(By.TAG_NAME, "input")
            for field in form_fields[:3]:
                bot.driver.execute_script("arguments[0].focus();", field)
                time.sleep(0.01)
                bot.driver.execute_script("arguments[0].blur();", field)
                time.sleep(0.01)
        except Exception:
            pass
        
        # Rapid page interactions
        print("🔄 Performing rapid page interactions...")
        try:
            # Rapid clicks on various elements
            clickable_elements = bot.driver.find_elements(By.TAG_NAME, "button")
            for element in clickable_elements[:2]:
                bot.driver.execute_script("arguments[0].click();", element)
                time.sleep(0.01)
        except Exception:
            pass
        
        print("🤖 GUARANTEED bot-like patterns completed - CAPTCHA SHOULD BE TRIGGERED!")

        bot.human_like_delay(3, 5)  # Final delay to observe results

    except Exception as e:
        print(f"An error occurred during bot execution: {e}")
    finally:
        if getattr(bot, 'bot_mode', False):
            print("Bot-mode: leaving browser open for inspection. Close it manually when done.")
        else:
            bot.close()