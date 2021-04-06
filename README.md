# tetr-analyzer
Tool to capture replays of versus Tetr.IO games and analyze statistics

# Installation
tetr-analyzer uses Selenium, a browser automation software. All required packages can be installed through pip with

```pip install -r requirements.txt```.

In addition, Selenium requires a webdriver to run. Currently, tetr-analyzer only supports Microsoft Edge. You can find a version of the webdriver on the [Microsoft Edge Developer](https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/#downloads) website that matches your version of Edge. Tetr-analyzer has not been tested on any version other than 89.0.774.68 (64-bit). Place the webdriver, named `msedgedriver.exe`, in the base directory.

Limited testing has also been done on Google Chrome, which similarly requires a Chrome webdriver, which can be found at the [Chromium](https://chromedriver.chromium.org/downloads) website. If no Edge webdriver is found, the script will search for `chromedriver.exe` in the base directory.

# Usage

Run

```python analyze.py -h```

to view all the optional command line arguments. The analyzer accepts a list of arguments which are either paths to a custom multiplayer game (`.ttrm` extension), replay IDs/links, or `?` for random top 100 games. If a non-cached custom game is specified, you will have to drag the file into the browser when prompted. Replay IDs must be preceded by "r:" or be a Tetr.IO URL, and only long replay IDs are currently supported. Each `?` argument will prompt the script to choose another random game.

After analysis, statistics will be computed over all the provided games, and displayed for each player.

Use `-v` for verbose output (recommended), and `-m` to enable manually config upload, which tends to be faster than letting the script set the settings every time. You will have to drag `config.ttc` into the browser when prompted in the command line.

Cached replay captures will be stored in the `cache/` folder, and will be used on subsequent calls to the script.