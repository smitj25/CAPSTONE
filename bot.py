import time
import random
import tempfile
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.edge.service import Service as EdgeService
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager

class HumanoidBot:
    def __init__(self, browser='chrome', driver_path=None, headless=False, bot_mode=False, delay_scale=1.2):
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
        self.bot_mode = bot_mode
        self.delay_scale = max(0.3, float(delay_scale))  # Minimum 0.3x speed for quick triggering
        self.driver = self._setup_driver(driver_path, headless)
        self.actions = ActionChains(self.driver)
        print(f"Bot initialized with {self.browser} browser (delay_scale: {self.delay_scale}x).")

    def _setup_driver(self, driver_path, headless):
        """Sets up the appropriate WebDriver based on browser choice."""
        if self.browser == 'chrome':
            options = webdriver.ChromeOptions()
            # Create a unique user data directory to avoid conflicts
            temp_dir = tempfile.mkdtemp()
            options.add_argument(f"--user-data-dir={temp_dir}")
            # Anti-detection options
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            if headless:
                options.add_argument('--headless')
            
            if driver_path:
                service = ChromeService(driver_path)
            else:
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
            else:
                service = FirefoxService(GeckoDriverManager().install())
            return webdriver.Firefox(service=service, options=options)
                
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
            else:
                service = EdgeService(EdgeChromiumDriverManager().install())
            return webdriver.Edge(service=service, options=options)
                
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
        self.human_like_delay(2, 4)  # Initial delay to load page

    def visit_local_file(self, html_file_name):
        """Navigates to the specified local HTML file (kept for backward compatibility)."""
        import os
        file_path = os.path.abspath(html_file_name).replace("\\", "/")
        url = f"file:///{file_path}"
        print(f"Visiting local file: {url}")
        self.driver.get(url)
        self.human_like_delay(2, 4)

    def visit_url(self, url):
        """
        Navigates to any URL.
        Args:
            url (str): The complete URL to visit
        """
        print(f"Visiting URL: {url}")
        self.driver.get(url)
        self.human_like_delay(2, 4)

    def human_like_delay(self, min_seconds, max_seconds):
        """Introduces a random delay to simulate human pauses."""
        if self.bot_mode:
            delay = random.uniform(0.0, 0.02)
        else:
            # Reduced delays for quicker CAPTCHA triggering
            base_delay = random.uniform(min_seconds, max_seconds) * self.delay_scale
            
            # 5% chance of a shorter "thinking" pause (reduced from 10%)
            if random.random() < 0.05:
                thinking_pause = random.uniform(0.5, 1.5) * self.delay_scale  # Reduced from 2.0-5.0
                delay = base_delay + thinking_pause
                print(f"[THINKING] Quick thinking pause: {delay:.2f} seconds...")
            else:
                delay = base_delay
                print(f"Pausing for {delay:.2f} seconds...")
        time.sleep(delay)

    def move_to_element_human_like(self, element, steps=50, max_offset=5):
        """
        Moves the mouse to an element with human-like, slightly erratic movements.
        Args:
            element: The WebElement to move to.
            steps (int): Number of small steps for the movement.
            max_offset (int): Maximum random offset from the direct path in pixels.
        """
        print(f"Moving mouse towards element: {element.tag_name} with ID: {element.get_attribute('id')}...")
        try:
            # Get the center of the element
            element_location = element.location
            element_size = element.size
            target_x = element_location['x'] + element_size['width'] / 2
            target_y = element_location['y'] + element_size['height'] / 2

            # Get current mouse position (approximate, Selenium doesn't expose it directly)
            current_x = self.driver.execute_script("return window.innerWidth / 2;")
            current_y = self.driver.execute_script("return window.innerHeight / 2;")

            # If it's the very first move, initialize position tracking
            if not hasattr(self, '_last_mouse_pos'):
                self.actions.move_by_offset(current_x, current_y).perform()
                self._last_mouse_pos = {'x': current_x, 'y': current_y}
            else:
                current_x = self._last_mouse_pos['x']
                current_y = self._last_mouse_pos['y']

            # Calculate total distance for each axis
            dx = target_x - current_x
            dy = target_y - current_y

            for i in range(steps):
                # Calculate intermediate target for this step
                inter_x = current_x + (dx / steps) * (i + 1)
                inter_y = current_y + (dy / steps) * (i + 1)

                # Add a small random offset to simulate human jitter
                offset_x = random.randint(-max_offset, max_offset)
                offset_y = random.randint(-max_offset, max_offset)

                # Move to the calculated point with offset
                self.actions.move_by_offset(inter_x - current_x + offset_x, inter_y - current_y + offset_y).perform()
                self.human_like_delay(0.01, 0.05)  # Small delays between micro-movements

                # Update current position for the next step calculation
                current_x = inter_x + offset_x
                current_y = inter_y + offset_y

            self._last_mouse_pos = {'x': current_x, 'y': current_y}  # Store last position
            print("Mouse movement complete.")
        except Exception as e:
            print(f"Error during human-like mouse movement: {e}")
            self.actions.move_to_element(element).perform()  # Fallback to direct move
            # If fallback, update the last known position to the element's center
            self._last_mouse_pos = {'x': element_location['x'] + element_size['width'] / 2,
                                    'y': element_location['y'] + element_size['height'] / 2}

    def click_element_human_like(self, element, offset_range=5):
        """
        Clicks an element with a slight, random offset to simulate human imprecision.
        Args:
            element: The WebElement to click.
            offset_range (int): Max pixels to offset the click from the center.
        """
        print(f"Attempting human-like click on element: {element.tag_name} with ID: {element.get_attribute('id')}...")
        try:
            # First, move to the element
            self.move_to_element_human_like(element)

            # Introduce a slight delay before clicking
            self.human_like_delay(0.2, 0.8)

            # Get element's size to calculate random offset within its bounds
            width = element.size['width']
            height = element.size['height']

            # Calculate a random offset relative to the center of the element
            offset_x = random.uniform(-min(offset_range, width/2 - 1), min(offset_range, width/2 - 1))
            offset_y = random.uniform(-min(offset_range, height/2 - 1), min(offset_range, height/2 - 1))

            # Move to the element's center and then apply the offset relative to its center
            self.actions.move_to_element(element).move_by_offset(offset_x, offset_y).click().perform()
            print(f"Clicked with offset ({offset_x:.2f}, {offset_y:.2f}).")
            
            # Update last known mouse position after click
            element_location = element.location
            element_size = element.size
            self._last_mouse_pos = {'x': element_location['x'] + element_size['width'] / 2 + offset_x,
                                    'y': element_location['y'] + element_size['height'] / 2 + offset_y}

        except Exception as e:
            print(f"Error during human-like click: {e}")
            self.actions.click(element).perform()  # Fallback to direct click
            # Update last known mouse position after fallback click
            element_location = element.location
            element_size = element.size
            self._last_mouse_pos = {'x': element_location['x'] + element_size['width'] / 2,
                                    'y': element_location['y'] + element_size['height'] / 2}

        self.human_like_delay(0.5, 1.5)  # Delay after click

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
        print(f"Typing human-like into element: {element.tag_name} with ID: {element.get_attribute('id')}...")
        for char in text:
            element.send_keys(char)
            self.human_like_delay(0.05, 0.2)  # Delay between keystrokes
        print("Typing complete.")

    def get_collected_data(self):
        """Retrieves the collected interaction data from the page."""
        try:
            output_element = self.driver.find_element(By.ID, "output")
            return output_element.text
        except Exception as e:
            print(f"Could not retrieve output data: {e}")
            return "No data collected or output element not found."

    def wait_for_element(self, by, value, timeout=10):
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

    def wait_for_clickable_element(self, by, value, timeout=10):
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
    
    bot = HumanoidBot(browser='chrome')  # Change this to test different browsers

    try:
        # Visit localhost:5173 (common Vite development server port)
        bot.visit_localhost_page("/", 5173)
        
        # Alternative ways to navigate:
        # bot.visit_localhost_page("/test", 5173)  # Visit specific path
        # bot.visit_url("http://localhost:3000")   # Visit different port
        # bot.visit_local_file("test.html")        # Still works for local files

        # Wait for and interact with elements (with better error handling)
        target_area = bot.wait_for_element(By.ID, "targetArea")
        if target_area:
            bot.click_element_human_like(target_area)
            bot.human_like_delay(1, 2)

        action_button = bot.wait_for_clickable_element(By.ID, "actionButton")
        if action_button:
            bot.click_element_human_like(action_button)
            bot.human_like_delay(1, 2)

        text_field = bot.wait_for_element(By.ID, "textField")
        if text_field:
            bot.type_human_like(text_field, "This is a test message from the multi-browser bot.")
            bot.human_like_delay(1, 2)

        # Scroll with random amount
        bot.scroll_human_like(random.randint(100, 300))
        bot.human_like_delay(1, 2)

        # Retrieve and print the collected data
        collected_data = bot.get_collected_data()
        print("\n--- Collected Bot Interaction Data ---")
        print(collected_data)
        print("--------------------------------------")

        bot.human_like_delay(3, 5)  # Final delay to observe results

    except Exception as e:
        print(f"An error occurred during bot execution: {e}")
    finally:
        bot.close()

