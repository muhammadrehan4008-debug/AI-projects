# DecoBot - Rule Based Chatbot

A simple rule-based chatbot built in Python using the Input -> Process -> Output
model. No machine learning here - just a dictionary lookup that matches exact
user input to a predefined response.

## Overview

DecoBot takes text input from the user, cleans it up (lowercase + trim
whitespace), and looks it up in a dictionary of known phrases. If a match is
found it returns the paired response, otherwise it falls back to a default
"I don't understand" message. The loop keeps running until the user types
`exit`, `quit`, or `q`.

## How it works

1. **Input** - user types a message, which gets lowercased and stripped of
   extra whitespace
2. **Process** - the cleaned message is looked up in a `responses` dictionary
   (O(1) lookup, no long if-elif chains)
3. **Output** - the matched response is printed, or a fallback message if
   nothing matches

## Features

- Greetings (hello, hi, hey)
- Farewells (bye, goodbye)
- Bot info (what are you, who are you, your name)
- Help command listing available phrases
- A joke
- Basic AI/ML definitions (what is ai, what is ml)
- Motivational responses (motivate me, i am stuck)
- Exit commands (exit, quit, q)

## Tech stack

- Python 3 (no external libraries required)

## How to run

```bash
python3 decobot.py
```

Type any of the known phrases (see `help` command in-app for the full list),
or type `exit` to quit.

## Example

```
You: hello
DecoBot: Hey there! I'm DecoBot. How can I help you today?

You: what is ai
DecoBot: AI stands for Artificial Intelligence - teaching machines to simulate
human decision-making through logic and data.

You: exit
DecoBot: Goodbye! Session terminated.
```

## Limitations

This is a rule-based system - it only responds to exact phrase matches (after
lowercasing/trimming). It doesn't understand synonyms, typos, or partial
matches. A natural next step would be adding fuzzy matching or keyword-based
matching instead of exact lookups.
