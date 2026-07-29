# Project 1 - Rule Based Chatbot
# Basic input -> process -> output flow, no ML involved, just a lookup table

responses = {
    # greetings
    "hello": "Hey there! I'm DecoBot. How can I help you today?",
    "hi": "Hi! Great to see you. What's on your mind?",
    "hey": "Hey! What can I do for you?",

    # farewells
    "bye": "Goodbye! Keep building cool things.",
    "goodbye": "See you later! Come back anytime.",

    # about the bot
    "what are you": "I'm DecoBot, a rule-based chatbot built at DecodeLabs.",
    "who are you": "I'm DecoBot, your first AI project. Simple but powerful!",
    "your name": "My name is DecoBot. Nice to meet you!",

    # help
    "help": "You can say: hello, bye, what are you, tell me a joke, "
            "what is ai, or just chat. Type 'exit' to quit.",

    # fun stuff
    "tell me a joke": "Why do programmers prefer dark mode? Because light attracts bugs!",
    "how are you": "I'm running at 100% efficiency - no bugs detected (yet).",

    # basic AI/ML knowledge
    "what is ai": "AI stands for Artificial Intelligence - teaching machines "
                  "to simulate human decision-making through logic and data.",
    "what is ml": "Machine Learning is a subset of AI where models learn "
                  "patterns from data instead of following hard-coded rules.",

    # motivation
    "motivate me": "Every expert was once a beginner. Keep coding!",
    "i am stuck": "Stuck? Good, that means you're learning. Break the problem "
                  "into smaller pieces and tackle one at a time.",
}

FALLBACK = "I don't understand that yet. Type 'help' to see what I know."
EXIT_COMMANDS = {"exit", "quit", "q"}


def get_clean_input():
    """Take raw input from the user and normalize it (lowercase, trimmed)."""
    raw = input("You: ")
    return raw.lower().strip()


def get_response(clean_input):
    """Look up the cleaned input in the responses dict, fallback if not found."""
    return responses.get(clean_input, FALLBACK)


def main():
    print("=" * 50)
    print("  DecoBot - Rule Based Chatbot")
    print("  Type 'exit' to quit.")
    print("=" * 50)

    while True:
        user_input = get_clean_input()

        if user_input in EXIT_COMMANDS:
            print("DecoBot: Goodbye! Session terminated.")
            break

        reply = get_response(user_input)
        print(f"DecoBot: {reply}\n")


if __name__ == "__main__":
    main()
