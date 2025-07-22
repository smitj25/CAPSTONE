import time
import random
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os # Import os module to handle file paths
from selenium.webdriver.chrome.service import Service

class HumanoidBot:
    def __init__(self, driver_path='chromedriver'):
        """
        Initializes the Selenium WebDriver.
        Args:
            driver_path (str): Path to your WebDriver executable (e.g., 'chromedriver').
        """
        options = webdriver.ChromeOptions()
        # Optional: Add some options to make it less detectable (though not foolproof)
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        # options.add_argument('--start-maximized') # Maximize window
        # Use Service object for driver path (Selenium 4+)
        service = Service(driver_path)
        self.driver = webdriver.Chrome(service=service, options=options)
        self.actions = ActionChains(self.driver)
        print("Bot initialized.")

    def visit_local_page(self, html_file_name):
        """Navigates to the specified local HTML file."""
        # Get the absolute path to the HTML file
        file_path = os.path.abspath(html_file_name).replace("\\", "/")
        url = f"file:///{file_path}"
        print(f"Visiting local file: {url}")
        self.driver.get(url)
        self.human_like_delay(2, 4) # Initial delay to load page

    def human_like_delay(self, min_seconds, max_seconds):
        """Introduces a random delay to simulate human pauses."""
        delay = random.uniform(min_seconds, max_seconds)
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
            # We'll assume the starting point is the center of the viewport for more realistic movement
            # This is still an approximation, but better than (0,0)
            current_x = self.driver.execute_script("return window.innerWidth / 2;")
            current_y = self.driver.execute_script("return window.innerHeight / 2;")

            # If it's the very first move, or you want to ensure it starts from somewhere
            # you can move to an initial random point first.
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
                self.human_like_delay(0.01, 0.05) # Small delays between micro-movements

                # Update current position for the next step calculation
                current_x = inter_x + offset_x
                current_y = inter_y + offset_y

            self._last_mouse_pos = {'x': current_x, 'y': current_y} # Store last position
            print("Mouse movement complete.")
        except Exception as e:
            print(f"Error during human-like mouse movement: {e}")
            self.actions.move_to_element(element).perform() # Fallback to direct move
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
            # First, move to the element (can reuse human_like_move)
            self.move_to_element_human_like(element)

            # Introduce a slight delay before clicking
            self.human_like_delay(0.2, 0.8)

            # Get element's size to calculate random offset within its bounds
            width = element.size['width']
            height = element.size['height']

            # Calculate a random offset relative to the center of the element
            # Ensure the offset keeps the click within the element's bounds
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
            self.actions.click(element).perform() # Fallback to direct click
            # Update last known mouse position after fallback click
            element_location = element.location
            element_size = element.size
            self._last_mouse_pos = {'x': element_location['x'] + element_size['width'] / 2,
                                    'y': element_location['y'] + element_size['height'] / 2}

        self.human_like_delay(0.5, 1.5) # Delay after click

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
            self.human_like_delay(0.05, 0.2) # Small delay between scroll steps
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
            self.human_like_delay(0.05, 0.2) # Delay between keystrokes
        print("Typing complete.")

    def get_collected_data(self):
        """Retrieves the collected interaction data from the page."""
        try:
            output_element = self.driver.find_element(By.ID, "output")
            return output_element.text
        except Exception as e:
            print(f"Could not retrieve output data: {e}")
            return "No data collected or output element not found."

    def close(self):
        """Closes the browser."""
        print("Closing bot.")
        self.driver.quit()

# --- Example Usage ---
if __name__ == "__main__":
    bot = HumanoidBot(driver_path='chromedriver.exe')

    try:
        # Visit the local HTML file
        bot.visit_local_page("test.html")

        # Interact with the elements
        target_area = WebDriverWait(bot.driver, 10).until(
            EC.presence_of_element_located((By.ID, "targetArea"))
        )
        bot.click_element_human_like(target_area)
        bot.human_like_delay(1, 2)

        action_button = WebDriverWait(bot.driver, 10).until(
            EC.element_to_be_clickable((By.ID, "actionButton"))
        )
        bot.click_element_human_like(action_button)
        bot.human_like_delay(1, 2)

        text_field = WebDriverWait(bot.driver, 10).until(
            EC.presence_of_element_located((By.ID, "textField"))
        )
        bot.type_human_like(text_field, "This is a test message from the bot.")
        bot.human_like_delay(1, 2)

        bot.scroll_human_like(random.randint(100, 300))
        bot.human_like_delay(1, 2)

        # Retrieve and print the collected data
        collected_data = bot.get_collected_data()
        print("\n--- Collected Bot Interaction Data ---")
        print(collected_data)
        print("--------------------------------------")

        bot.human_like_delay(3, 5) # Final delay to observe results

    except Exception as e:
        print(f"An error occurred during bot execution: {e}")
    finally:
        bot.close()