from slackclient import SlackClient

from config import SLACK_TOKEN


class Slack:
    sc = None

    def __init__(self):
        self.sc = SlackClient(SLACK_TOKEN)

    def send(self, msg):
        self.sc.api_call(
            "chat.postMessage",
            channel="#ades-bot",
            text=msg
        )
        print(msg)