# Browser control through selenium
# https://www.selenium.dev/

import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import sys
import os

JS_DROP_FILE = """
    var target = arguments[0],
        offsetX = arguments[1],
        offsetY = arguments[2],
        document = target.ownerDocument || document,
        window = document.defaultView || window;

    var input = document.createElement('INPUT');
    input.type = 'file';
    input.onchange = function () {
      var rect = target.getBoundingClientRect(),
          x = rect.left + (offsetX || (rect.width >> 1)),
          y = rect.top + (offsetY || (rect.height >> 1)),
          dataTransfer = { files: this.files };

      ['dragenter', 'dragover', 'drop'].forEach(function (name) {
        var evt = document.createEvent('MouseEvent');
        evt.initMouseEvent(name, !0, !0, window, 0, 0, 0, x, y, !1, !1, !1, !1, 0, null);
        evt.dataTransfer = dataTransfer;
        target.dispatchEvent(evt);
      });

      setTimeout(function () { document.body.removeChild(input); }, 25);
    };
    document.body.appendChild(input);
    return input;
"""

class Browser:

    
    def __init__(self, *, speedup=2, manual_config=False, verbose=False):
        self._open = False
        self.driver = None
        assert speedup in (1, 2, 5, 10)
        self.speedup = speedup
        self.manual_config = manual_config
        self.verbose = verbose

    def _log(self, *args, **kwargs):
        if self.verbose:
            print("[Browser] ", *args, **kwargs)

    def init(self):
        if self._open: return
        self._log("Opening Selenium-driven browser")
        try:
            if os.path.exists("msedgedriver.exe"):
                self.driver = webdriver.Edge("msedgedriver.exe")
            else:
                self.driver = webdriver.Chrome("chromedriver.exe")
        except Exception as e:
            print(e)
            print("Make sure you have a webdriver in the working directory")
            sys.exit(1)
        self.driver.set_window_size(1100, 700)
        self.driver.get("https://tetr.io/")
        # Login
        self._log("Logging in")
        # project-t
        # xyn123
        self.wait_action("#entry_username", "send_keys", "project-t")
        self.wait_action("#entry_button", "click")
        self.wait_action("#login_password", "send_keys", "xyn123")
        self.wait_action("#login_button", "click")
        # Run resize script
        padding = self.driver.execute_script(
            "return [window.outerWidth-window.innerWidth, window.outerHeight-window.innerHeight];"
        )
        self.driver.set_window_size(1084 + padding[0], 570 + padding[1])
        time.sleep(2)
        # Close notifications if they are up
        for _ in range(2):
            self.wait_action(".notification", "click", timeout=2)
        self.wait_action(".patchnotes .oob_button.pri", "click", timeout=10)
        # Change config settings
        if self.manual_config:
            self._log("Drag 'config.ttc' into the browser...")
            self.wait_action("#dialogs div.oob_button.sec", "click", timeout=1 * 60)
        else:
            self._log("Setting correct configuration...")

        self.set_config()
        # Refresh and re-login
        self._log("Refreshing page to confirm settings")
        self.driver.refresh()
        time.sleep(0.25)
        self.wait_action("#return_button", "click")
        self._open = True

    def finish(self):
        if self.driver:
            self._log("Closing browser")
            self.driver.close()
        self._open = False
        self.driver = None

    def get(self, replayid, rounds, framecounts):
        self.init()
        self.open_replay(replayid)
        for round, framecount in zip(rounds, framecounts):
            self._log(f"Recording round {round + 1}")
            self.capture_replay(round, framecount)
            yield self.all_replay_data()
        self.close_replay()

    def wait_action(self, selector, action, *args, timeout=5, index=0, **kwargs):
        success = False
        timer = time.time()
        while not success and time.time() - timer < timeout:
            element = self.driver.find_elements_by_css_selector(selector)
            if element:
                try:
                    getattr(element[index], action)(*args, **kwargs)
                    success = True
                except Exception:
                    pass
        return success
    def drag_and_drop_file(self, drop_target, path):
        path = os.path.abspath(path)
        driver = drop_target.parent
        file_input = driver.execute_script(JS_DROP_FILE, drop_target, 0, 0)
        file_input.send_keys(path)
    def set_config(self):
        target = self.driver.find_element_by_id("victoryview")
        self.drag_and_drop_file(target, "config.ttc")
        self.wait_action("#dialogs > div > div > div.oob_button.flex-item.sec", "click")
        return
        for element in [
                "#sig_config",
                "h1[title='Change the way TETR.IO sounds']",
                "#volume_disable",
                "h1[title='Change the way TETR.IO functions']",
                "#video_actiontext_off",
                "#video_spin",
                "#video_spikes",
                "#video_kos",
                "#video_fire",
                "#video_siren",
                "h1[title='Change the way TETR.IO looks']"
            ]:
            self.wait_action(element, "click")
        for element, value in [
                ("#video_bounciness_field", 0),
                ("#video_shakiness_field", 0),
                ("#video_gridopacity_field", 0),
                ("#video_boardopacity_field", 1),
                ("#video_shadowopacity_field", 0),
                ("#video_background_field", 0),
                ("#video_particles_field", 0.1),
                ("#video_pieceflash_field", 0)
            ]:
            try:
                self.wait_action(element, "click")
                time.sleep(0.05)
                self.driver.execute_script(
                    "arguments[0].setAttribute('value', arguments[1])",
                    self.driver.find_element_by_id("request_number"),
                    str(value)
                )
                time.sleep(0.05)
                self.wait_action("#request_number_submit", "click")
                time.sleep(0.05)
            except Exception as e:
                print(f"Errored on {element}")
                raise e
        self.wait_action("#video_graphics_minimal", "click")

    def open_replay(self, replayid=None):
        self.wait_action("#sig_channel", "click")
        if not replayid.endswith(".ttrm"):
            self._log(f"Opening replay: r:{replayid}")
            self.wait_action("#tetra_find", "clear")
            self.wait_action("#tetra_find", "send_keys", "r:" + replayid)
            self.wait_action("#tetra_find", "send_keys", "\n")
        else:
            target = self.driver.find_element_by_id("victoryview")
            self.drag_and_drop_file(target, replayid)
            WebDriverWait(self.driver, 2 * 60).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#multilogview:not(.hidden)"))
            )

    def close_replay(self):
        self._log("Closing replay")
        self.wait_action("#back", "click")
        self.wait_action("#back", "click")

    def capture_replay(self, roundnum, framecount):
        self.wait_action(".multilog_result.scroller_block.zero", "click", index=roundnum)
        self.wait_action(f"#replaytools_button_{self.speedup}x", "click")
        self.wait_action("#replaytools_button_playpause", "click")
        time.sleep(5)
        for filename, tag in [
                ("scripts/capture_grid.js", "grid"),
                ("scripts/capture_next.js", "next"),
                ("scripts/capture_hold.js", "hold")
            ]:
            for side in [0, 1]:
                self.run_script(filename, {
                    "[MAX_FRAMES]": framecount,
                    "[PLAYER_SIDE]": side,
                    "[DATA_TAG]": f"{tag}-{side}"
                })
        self.wait_action("#replaytools_button_playpause", "click")

    def run_script(self, filename, replace={}):
        with open(filename) as f:
            script = f.read()
        for k, v in replace.items():
            script = script.replace(k, str(v))
        self.driver.execute_script(script)

    def get_replay_data(self, tag):
        try:
            dataElement = WebDriverWait(self.driver, 6 * 60).until(
                EC.presence_of_element_located((By.ID, f"captured-{tag}"))
            )
            return json.loads(dataElement.get_attribute("innerText"))
        except Exception as err:
            self.finish()
            raise err

    def all_replay_data(self):
        data = ()
        for side in [0, 1]:
            sidedata = {}
            for tag in ["grid", "next", "hold"]:
                sidedata[tag] = self.get_replay_data(f"{tag}-{side}")
            data = data + (sidedata,)
        return data


