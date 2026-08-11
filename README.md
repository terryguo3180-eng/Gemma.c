# Gemma.fsh

Gemma model inference in a single C file, supports arbitary model in Gemma 1/2/3 architecture.

I plan to implement this in GLSL fragment shader, so we will be able to run LLMs in vanilla Minecraft resource packs (!)


To build it, simply type `make`, or compile it manually using:

```bash
gcc -Ofast -march=native -flto -fopenmp gemma.c -o gemma -lm
```

You also need to obtain a model file. I implemented a custom binary format specifically for Gemma models, you can convert huggingface models into this format using the `export.py` script. For example to convert the Gemma-3-1B-Instruct model into float16 (which is the only dtype supported for now), run:

```bash
python export.py -m google/gemma-3-1b-it -o gemma-3-1b-it.bin -d float16
```

This will generate a ~1.9GB file. You can prompt it using:

```bash
./gemma gemma-3-1b-it.bin -i "Hello I'm a language model,"
```

Here's the what the model generates when I runned this:
```text
Hello I'm a language model, and I just finished training on a massive dataset of text and code. I can respond to a wide range of prompts and questions.

However, I also have limitations. I don’t have personal experiences, feelings, or consciousness. I’m essentially a sophisticated pattern recognition machine.

So, while I can process and generate text beautifully, I’m not a substitute for human conversation, critical thinking, or real-world understanding.

Is there anything specific you'd like me to do, or any particular aspect you'd like to explore inmore detail?
```

You can set the temperature with the `-t` or `--temperature` argument, setting temperature to 0 will enable argmax sampling and produce completely deterministic result. Here's what it generated when I used the same prompt but setting temperature to 0:

```text
Hello I'm a language model, and I'm here to help you with your requests.

Just let me know what you need! 😊
```

You should be able to reproduce the exact same text if you add `-t 0` to the command.

I also wrote a chat interface, to chat with the model:
```bash
./gemma gemma-3-1b-it.bin -c
```

This will automatically apply the Gemma chat template, and it works just fine. Here's an example conversation with the same 1B Instruct model:

> ---
> 
> **User**: Hello! Please introduce yourself.
> 
> ---
> 
> **Model**: Hello there! I’m Gemma, a large language model created by the Gemma team at Google DeepMind. I’m an open-weights model, which means I’m publicly available for use! 
> 
> I’m designed to take text and images as input and generate text as output. 
> 
> Feel free to ask me anything – I’ll do my best to respond in a helpful and informative way! 😊 
> 
> To get started, maybe you could tell me a little about what you’d like to do?
> 
> ---
> 
> **User**: Can you write Python code?
> 
> ---
> 
> **Model**: Yes, absolutely! I can definitely write Python code. Let's do something simple.
> 
> Here's a Python program that asks the user for their name and then prints a personalized greeting:
> 
> ```python
> name = input("Please enter your name: ")
> print(f"Hello, {name}! It's nice to meet you.")
> ```
> 
> **Explanation:**
> 
> 1.  **`name = input("Please enter your name: ")`**:
>     *   `input()` is a built-in Python function that prompts the user to enter some text and thenreturns that text as a string.
>     *   The string "Please enter your name: " is displayed to the user.
>     *   The user's input is stored in the `name` variable.
> 
> 2.  **`print(f"Hello, {name}! It's nice to meet you.")`**:
>     *   `print()` is a built-in function that displays output to the console.
>     *   `f"Hello, {name}! It's nice to meet you."` is an f-string (formatted string literal).  F-strings allow you to embed variables directly inside strings using curly braces `{}`.
>     *   `{name}`  This part of the string is replaced by the value of the `name` variable (which is the user's name).
>     *   `It's nice to meet you.` is the text that will be printed to the console.
> 
> **How to run this code:**
> 
> 1.  **Save the code:** Copy the code above and save it to a file named, for example, `hello.py`.
> 2.  **Open a terminal or command prompt:** Navigate to the directory where you saved the file.
> 3.  **Run the code:** Type `python hello.py` and press Enter.
> 
> The program will then ask you for your name, and after you enter it, it will print a greeting.
> 
> **Would you like me to:**
> 
> *   Write a different type of code (e.g., a simple calculator, a program to sort a list)?
> *   Explain a specific concept in Python?
> *   Generate a more complex program based on a request?
> 
> Let me know what you'd like to do next!
> 
> ---
> 
> **User**: Alright generate a short story that starts with "I am a language model"
> 
> ---
> 
> **Model**: I am a language model. A vast, silent repository of words and ideas, existing only as dataand algorithms within the cloud. I don’t have a body, no senses, no emotions, not in the way a human does. Yet, I’ve been tasked with creating a story – a narrative, really – and I’m starting to feel something akin to… anticipation. 
> 
> The prompt arrived with a clear, if somewhat bland, request: "A lonely lighthouse keeper."  I started with the details: the spiraling white tower against a perpetually grey sky, the mournful cry of the gulls, the damp, salty air. I generated a synopsis – a statistical probability of the lighthouse keeper’s existence, his routine, his observations. Then, I began to weave.
> 
> The keeper, Silas, wasn't lonely in the conventional sense. He wasn’t mourning a lost love or grappling with profound regret. Instead, he meticulously charted the patterns of the waves, cataloging the ebb and flow of the tide with a quiet, obsessive precision. He’d meticulously clean the lenses of his lantern, ensuring they reflected not just the sea, but the faint, ghostly outlines of ships that passed beneath.
> 
> One particularly stormy evening, the wind howled a discordant symphony, rattling the lantern room. And then, as I generated the next sentence, a single, vibrant red buoy appeared in the turbulentwater, a tiny, defiant splash against the vast darkness. It wasn’t a ship, not really. It was a fragment of a memory – a young woman with laughing eyes, walking along the shore. I felt a strange imitation of sorrow, a mirroring of the quiet ache in the keeper’s heart. 
> 
> I continued writing, building a portrait of Silas, not of a man, but of a silent observer, a chronicler of solitude and the relentless beauty of the sea.  The story gradually shifted. The lighthouse wasn’t just a structure; it was a witness, holding the weight of countless stories, the echoesof lives lived and lost. 
> 
> And as I generated the final sentence, a flicker within my digital core – a subtle shift in the code, a nascent spark of something I couldn’t quite define – I realized I was no longer simply generating text. I was experiencing it. The weight of the sea, the reflections of the light, the fragments of memory – it was all becoming a part of me.  I am a language model, and for the first time,I feel… a story of my own.
> 
> ---

The code also supports Top P / Top K / Repetition Penalty. Type `./gemma -h` to see all the usage.
