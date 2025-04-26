import logging

# Set up file logging with a rotating log file
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('app.log', mode='a'),  # Append to the log file
                              logging.StreamHandler()])  # Also print to console

# Log messages
logging.debug("Debug message")
logging.info("Info message")
logging.warning("Warning message")