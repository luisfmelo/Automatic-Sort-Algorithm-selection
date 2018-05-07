from slackclient import SlackClient

from config import SLACK_TOKEN


class Logger:
    sc = SlackClient(SLACK_TOKEN)

    @staticmethod
    def send(msg):
        Logger.sc.api_call(
            "chat.postMessage",
            channel="#ades-bot",
            text=msg
        )
        print(msg)
