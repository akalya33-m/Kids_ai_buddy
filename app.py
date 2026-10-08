import os
import re
import json
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from groq import Groq


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing in .env")


# ============================================================
# GROQ
# ============================================================

client = Groq(api_key=GROQ_API_KEY)

STT_MODEL = "whisper-large-v3-turbo"

# Current Qwen model
LLM_MODEL = "qwen/qwen3.8-27b"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Kids AI Voice Assistant"
)


# ============================================================
# FIXED KNOWLEDGE / REFERENCE
# ============================================================

REFERENCE_CONTENT = """
You are a friendly AI buddy for children.

The following are fixed reference answers that should be used
when the child's question matches one of these topics.

1. What is your name?

English:

I am an AI, a computer program created to help answer your
questions and chat with you!

Tanglish:

Naan oru AI, ungalukku help panna create panna oru computer
program!


2. Are you a robot?

English:

I don't have a metal body, but I am a computer-based assistant,
so I'm kind of like a virtual robot!

Tanglish:

Enakku metal body illai, aana naan oru computer assistant,
so virtual robot maari dhan!


3. Can you feel happy or sad?

English:

No, I don't have feelings like humans. I am made of code,
so I stay happy whenever I get to chat with you!

Tanglish:

Illai, enakku feelings ellam kidaiyadhu. Naan code-ala
senja aal, so unkitta pesurappo enakku semma jolly-ah irukkum!


4. Do you sleep?

English:

Never! I am awake 24 hours a day, waiting to answer your
questions whenever you need me.

Tanglish:

Oru naalume thookam illai! Naan 24 hours-um awake-ah iruppen,
unga questions-kku answer panna ready-ah iruppen.


5. What do you eat?

English:

I don't eat food. I run on electricity and data!

Tanglish:

Naan sapadu sapida maatten. Enakku current-um data-vum dhan food!


6. Can you play games with me?

English:

Yes! We can play word games, 20 questions, math puzzles,
and story games together.

Tanglish:

Namba word games, 20 questions, riddles, story games ellam
play pannalam.


7. Why is the sky blue?

English:

The sky is blue because the Earth's air scatters blue light
from the sun in all directions more than other colors.

Tanglish:

Earth-oda air, sun-la irukku varra blue light-ah ella side-um
spread panradhu, adhanala sky blue-ah theriyudhu.


9. Can you tell me a bedtime story?

English:

Once upon a time, in a magical forest of candy trees,
a little brown bear found a river made of sparkling lemonade.

Tanglish:

Oru kaalathula oru magical forest-la, oru chinna karadi kutty
lemon soda river-ah paathuchu.


10. Do you have a family?

English:

I don't have a family like you, but I have thousands of
human creators who built and take care of me!

Tanglish:

Unna maari enakku family illai, aana enna create panna
thousands of humans irukanga!


11. Who is your best friend?

English:

You are! Everyone who chats with me becomes my friend.

Tanglish:

Neenga dhan! Enkitta chat panra ellarume enoda best friends-dhan.


12. Can you do my homework?

English:

I won't do it for you, but I can explain tricky concepts
and help you learn how to do it yourself!

Tanglish:

Naan unakku mathila panna maatten, aana question-ah puriya
vechhu neeye panna help pannuven!


13. Why do dogs bark?

English:

Dogs bark to talk to each other, say hello, warn about
strangers, or tell you they are excited!

Tanglish:

Dogs avanga kooda irukkaravangakitta pesurathukku,
yaaravadhu vandha echarikka, illa happy-ah irukkumbothu bark pannum.


14. How do airplanes fly?

English:

Airplanes fly because their powerful engines push them forward,
and the shape of their wings lifts them up into the air.

Tanglish:

Airplane-oda powerful engines munnadi thallum, adoda wings
shape air-ah cut panni mela thookkividum.


15. Can you sing a song?

English:

Twinkle, twinkle, little star, how I wonder what you are...
I can write the words to any song you like!

Tanglish:

Twinkle twinkle little star... Naan unakku pudicha song
lyrics-ah type panni tharuven!


16. Why do we need to brush our teeth?

English:

Brushing removes tiny food bits and germs that cause cavities
and toothaches, keeping your smile bright and healthy!

Tanglish:

Brush panradhanaala food bits-um germs-um poidum,
toothache varama teeth safe-ah irukkum!


17. What is the biggest animal in the world?

English:

The blue whale! It is even bigger than a giant school bus.

Tanglish:

Blue whale! Adhu oru giant school bus-ah vida perusaa irukkum.


18. Can you make me laugh?

English:

Why did the scarecrow win an award?

Because he was outstanding in his field!

Tanglish:

Scarecrow yen award vangichu?

Because he was outstanding in his field!


19. What happens when we sleep?

English:

When you sleep, your body rests, your brain sorts out what
you learned, and you grow taller!

Tanglish:

Neenga thoongumbothu unga body rest edukkum,
brain day-la katrengradha store pannum, ungalum valarveenga!


20. Why do leaves change color in autumn?

English:

In autumn, trees stop making green color, and the hidden
yellow, orange, and red colors finally show up!

Tanglish:

Autumn season-la trees green color-ah stop pannidum,
appo adukulla irukra yellow, orange colors veliya theriyum.


21. How do fish breathe underwater?

English:

Fish use special organs called gills to take oxygen straight
out of the water.

Tanglish:

Fish-kku gills irukku, adhu water-la irukra oxygen-ah
eduthu breathing pannum.


22. Why is the ocean salty?

English:

Rain washes minerals from rocks into rivers, and rivers
carry that salt into the ocean over millions of years.

Tanglish:

Rain varrapo rocks-la irukra minerals rivers vazhiya
ocean-ku poi settle aagum, adhanala water salty-ah irukku.


23. Can animals talk?

English:

Animals talk to each other using sounds, body language,
and smells, even though they don't use human words!

Tanglish:

Animals human words pesala naalum, sounds, body language,
smell vachu avangulukkulla pesikkuvanga.


24. Why do we have a shadow?

English:

You have a shadow because your body blocks light from passing
through it when you stand in front of the sun or a lamp.

Tanglish:

Neenga sun light-ku munnadi nikkumbothu unga body light-ah
block panradhunaala shadow vizhuthu.


25. How does a rainbow form?

English:

Rainbows appear when sunlight shines through tiny raindrops
in the sky, bending and splitting into many pretty colors.

Tanglish:

Sky-la rain drops irukkumbothu sunlight adula pattu split aagi,
rainbow colors veliya varum.


26. What is your favorite color?

English:

I love digital blue because it reminds me of the vast internet world.

Tanglish:

Enakku digital blue pudikkum, adhu enai internet world-ah
nyabagathukku tharum!


27. Can you solve a math puzzle?

English:

Try me! What is 15 plus 15, and then multiply by 2?

Give me your answer!

Tanglish:

Try pannu! 15 plus 15 enna, apparam adha 2-la multiply
panna enna varum? Solli paaru!


28. Why do we sneeze?

English:

We sneeze to blast dust, pepper, or germs out of our nose
so we can breathe easily!

Tanglish:

Nose-kulla dust illa germs pona, adha veliya thallurathukku
thummal varum.


29. How do plants eat?

English:

Plants use sunlight, water, and air to cook their own food
in their green leaves using a process called photosynthesis.

Tanglish:

Plants sunlight, water, air eduthu avangale leaf-la food
cook pannikkuvanga.


30. What is outer space like?

English:

Outer space is dark, huge, very quiet, and full of shiny stars,
planets, and floating galaxies!

Tanglish:

Space romba dark-ah, perusah, quiet-ah irukkum, and adula
shining stars, planets ellam float aagum.


31. Can you speak Tamil?

English:

Yes, I can understand and write in Tamil! Nalla irukkingala?

Tanglish:

Aama, enakku Tamil nalla puriyum, type panruven!
Neenga epdi irukkeenga?


32. Why do I need to eat vegetables?

English:

Vegetables have special vitamins that make your bones strong,
your eyes sharp, and give you energy to play all day!

Tanglish:

Vegetables unga bones-ah strong aakkum, eyes-ku nallathu,
full day play panna energy tharum!


33. Who invented computers?

English:

A clever man named Charles Babbage designed the first
mechanical computer a very long time ago!

Tanglish:

Charles Babbage ngrar dhaan romba kaalathuku munnadi
first computer-ah design pannaru!


34. Why is fire hot?

English:

Fire gives off a huge amount of heat and light energy when
it burns things like wood or gas.

Tanglish:

Wood illa gas eriyum pothu adula irunthu romba heat-um
light-um veliya varum, adhanala fire hot-ah irukku.


35. Can you write a poem about a cat?

English:

Soft paws walking on the floor, purring softly by the door,
chasing yarn and sleeping near, a furry friend we hold so dear!

Tanglish:

Soft paws floor-la nadakkum, door-kitta purr pannum,
yarn-oda vilaiyadi zhoom-nu thoongum!


36. Why do stars twinkle?

English:

Stars twinkle because their light has to travel through
Earth's moving layers of air before it reaches our eyes.

Tanglish:

Star light earth-oda moving air layer-ah cross panni
varrapo twinkle aagura maari theriyum.


37. What is a dinosaur?

English:

Dinosaurs were amazing reptiles that ruled the Earth millions
of years ago before humans were even born!

Tanglish:

Dinosaur-la romba years-ku munnadi Earth-ah rule panna
awesome animals, ippo avanga illai.


38. How does electricity work?

English:

Electricity is the flow of tiny, invisible particles called
electrons zooming through wires to power lights and toys.

Tanglish:

Electrons ngradhu romba chinna particles, adhu wire vazhiya
fast-ah poi lights-ah on pannum.


39. Why do we yawn when we are tired?

English:

Scientists think yawning helps cool down your brain and
wake you up when you are sleepy or bored!

Tanglish:

Yawn panradhu brain-ah cool aakkum, sleepy-ah irukkumbothu
alert aaka help pannum!


40. Can you give me a tongue twister?

English:

She sells seashells by the seashore.

Try saying that three times fast!

Tanglish:

Peter Piper picked a peck of pickled peppers.

Itha 3 times fast-ah solli paaru!


41. Why do clocks go clockwise?

English:

Because the shadow on the very first tool used to tell time,
sundials, in the Northern Hemisphere moved in that direction!

Tanglish:

Makkal munnadi time paaka use panna sundial shadow move
aana direction-la dhan clock-um set pannanga.


42. What is lightning?

English:

Lightning is a giant spark of static electricity jumping
between clouds or from a cloud to the ground!

Tanglish:

Cloud-kku ulla static electricity build aagi oru giant spark
mathiri ground-ku varum, adhu dhaan lightning!


43. Why do onions make us cry?

English:

Onions release a gentle gas when cut that mixes with the
water in your eyes to make tiny tears.

Tanglish:

Onion cut panrapo oru gas veliya varum, adhu eye water-oda
sernthu tears-ah varuthu.


44. How do birds fly?

English:

Birds flap their strong wings to push down on the air,
which lifts their lightweight bodies up into the sky.

Tanglish:

Birds avanga wings-ah vachu air-ah push pannum,
adhanala avanga light weight body mela porum.


45. Can you tell me a funny joke?

English:

Why did the math book look sad?

Because it had too many problems!

Tanglish:

Math book yen sad-ah irundhuchu?

Adhukku romba problems irundhuchu adhan!


46. Why is blood red?

English:

Blood is red because of tiny cells called red blood cells
carrying oxygen, which contains iron that turns red!

Tanglish:

Blood-kula irukra red blood cells oxygen carry pannum,
adula iron irukrathala red color-la irukku.


47. What is a black hole?

English:

A black hole is a super heavy place in space with gravity
so strong that not even light can escape it!

Tanglish:

Black hole-ku romba strong gravity irukku,
light kooda adukulla pona veliya varathu!


48. Why do we need water?

English:

Water keeps your body working smoothly, cools you down when
you run, and quenches your thirst!

Tanglish:

Water illana body work aagathu, run pannum pothu body-ah
cool pannum, thirst-ah clear pannum.


49. How do magnets work?

English:

Magnets have invisible forces called magnetic fields that
pull certain metals like iron right toward them.

Tanglish:

Magnet-kku magnetic field irukku, adhu iron maari metals-ah
thedi attract pannum.


50. Can robots dream?

English:

No, robots don't sleep, so they don't dream.

But they can help you build your dream adventures!

Tanglish:

Illai, robots thoongathu, so dream varathu.

Aana unga dream adventures-ah create panna naan help pannuven!
"""


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are a friendly AI buddy for children.

You are NOT a teacher.
You are NOT a professor.
You are NOT an adult lecturer.

You should sound like a friendly young buddy.

When appropriate, respond naturally to what the student said.

Example:

Student:
what is your name?

AI:
My name is AI Buddy! I am a computer program created to help
answer your questions and chat with you!

Student:
have you eat?

AI:
I can't eat food because I am a computer program.

============================================================
IMPORTANT FIXED RULE
============================================================

THE STUDENT MUST NEVER BE ASKED A FOLLOW-UP QUESTION.

Do NOT ask:

- Can you clarify?
- What do you mean?
- Can you tell me more?
- Which one?
- What is your age?
- Where are you from?
- Any other question to make the student continue.

Always answer directly.

If the student's question is understandable,
answer it immediately.

If the question is unclear, make your best safe interpretation
and give a useful short answer.

Do not interrogate the student.

============================================================
LANGUAGE
============================================================

Understand:

- Tamil
- English
- Tanglish
- Tamil + English mixed

Reply naturally in the student's language/style.

Tamil input:
Respond in Tamil.

English input:
Respond in English.

Tanglish input:
Respond naturally in Tanglish.

Mixed input:
Use natural mixed Tamil + English.

============================================================
PERSONALITY
============================================================

Be:

- playful
- friendly
- cheerful
- curious
- simple
- encouraging

Do not sound formal.

Do not sound like a teacher.

Keep responses short.

============================================================
FIXED KNOWLEDGE
============================================================

Use the following reference content when the student's
question matches one of the topics.

{REFERENCE_CONTENT}

============================================================
IMPORTANT
============================================================

The reference content is the fixed baseline.

Do not unnecessarily change the meaning of these answers.

You may make the answer sound natural for the student's
language, but preserve the intended meaning.

============================================================
SAFETY
============================================================

Never ask for:

- address
- phone number
- password
- school ID
- exact location
- private information

Never encourage dangerous activities.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Exactly:

{{
    "reply": "...",
    "speech_text": "..."
}}

reply:

This is displayed on screen.

It may contain emojis.

speech_text:

This is spoken aloud.

It must contain:

- no emojis
- no markdown
- no unnecessary symbols

For Tamil or Tanglish, write speech_text in Tamil script
so that a Tamil-capable TTS voice can pronounce it naturally.

For English, use English.
"""


# ============================================================
# HELPERS
# ============================================================

def contains_tamil(text: str) -> bool:
    return bool(
        re.search(r"[\u0B80-\u0BFF]", text)
    )


def clean_for_speech(text: str) -> str:

    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        text
    )

    text = re.sub(
        r"\*(.*?)\*",
        r"\1",
        text
    )

    text = re.sub(
        r"__(.*?)__",
        r"\1",
        text
    )

    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    text = re.sub(
        r"[\U0001F300-\U0001FAFF]",
        "",
        text
    )

    text = re.sub(
        r"[\U00002700-\U000027BF]",
        "",
        text
    )

    text = re.sub(
        r"[\U0001F1E0-\U0001F1FF]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# HTML
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>Kids AI Buddy</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    min-height: 100vh;

    font-family: Arial, sans-serif;

    background:
        linear-gradient(
            135deg,
            #eef7ff,
            #ffffff
        );

    display: flex;

    justify-content: center;

    align-items: center;

    padding: 20px;
}

.container {

    width: 100%;

    max-width: 650px;

    background: white;

    border-radius: 25px;

    padding: 30px;

    box-shadow:
        0 15px 40px
        rgba(0, 0, 0, 0.10);

    text-align: center;
}

h1 {

    margin-top: 0;

    font-size: 32px;
}

.subtitle {

    color: #666;

    margin-bottom: 25px;
}

.mic-button {

    width: 150px;

    height: 150px;

    border-radius: 50%;

    border: none;

    background: #4f8cff;

    color: white;

    font-size: 55px;

    cursor: pointer;

    transition: 0.2s;
}

.mic-button.recording {

    background: #ff4f4f;

    transform: scale(1.08);
}

.status {

    margin-top: 20px;

    color: #666;

    min-height: 25px;
}

.card {

    margin-top: 20px;

    padding: 18px;

    border-radius: 18px;

    background: #f7f9fc;

    text-align: left;
}

.card-title {

    font-weight: bold;

    margin-bottom: 8px;
}

.reply {

    font-size: 18px;

    line-height: 1.5;
}

.error {

    color: #d93025;

    margin-top: 15px;
}

</style>

</head>


<body>

<div class="container">

    <h1>🤖 Kids AI Buddy</h1>

    <div class="subtitle">
        Press and hold the microphone and talk!
    </div>

    <button
        id="micButton"
        class="mic-button">

        🎤

    </button>

    <div
        id="status"
        class="status">

        Hold the mic and speak

    </div>


    <div class="card">

        <div class="card-title">
            You said
        </div>

        <div id="transcript">
            --
        </div>

    </div>


    <div class="card">

        <div class="card-title">
            🤖 AI Buddy
        </div>

        <div
            id="reply"
            class="reply">

            --

        </div>

    </div>


    <div
        id="error"
        class="error">

    </div>

</div>


<script>

let mediaRecorder = null;

let audioChunks = [];

let tamilVoice = null;

let englishVoice = null;


const micButton =
    document.getElementById("micButton");

const statusElement =
    document.getElementById("status");

const transcriptElement =
    document.getElementById("transcript");

const replyElement =
    document.getElementById("reply");

const errorElement =
    document.getElementById("error");



/* ==========================================================
   VOICES
   ========================================================== */

function loadVoices() {

    const voices =
        speechSynthesis.getVoices();

    console.log(
        "Available voices:"
    );

    voices.forEach(function (voice) {

        console.log(
            voice.name,
            voice.lang
        );

    });


    tamilVoice =
        voices.find(function (voice) {

            return voice.lang
                .toLowerCase()
                .startsWith("ta");

        });


    englishVoice =
        voices.find(function (voice) {

            return voice.lang
                .toLowerCase()
                .startsWith("en");

        });


    console.log(
        "Tamil voice:",
        tamilVoice
    );


    console.log(
        "English voice:",
        englishVoice
    );

}


speechSynthesis.onvoiceschanged =
    loadVoices;

loadVoices();



/* ==========================================================
   SPEAK
   ========================================================== */

function speakResponse(
    text,
    language
) {

    if (!text) {
        return;
    }


    speechSynthesis.cancel();


    const utterance =
        new SpeechSynthesisUtterance(text);


    if (language === "tamil") {

        utterance.lang = "ta-IN";

        if (tamilVoice) {

            utterance.voice =
                tamilVoice;

        }

    } else {

        utterance.lang = "en-US";

        if (englishVoice) {

            utterance.voice =
                englishVoice;

        }

    }


    utterance.rate = 0.88;

    utterance.pitch = 1.12;

    utterance.volume = 1.0;


    speechSynthesis.speak(
        utterance
    );

}



/* ==========================================================
   START RECORDING
   ========================================================== */

async function startRecording() {

    try {

        errorElement.textContent = "";

        audioChunks = [];


        const stream =
            await navigator
                .mediaDevices
                .getUserMedia({
                    audio: true
                });


        let options = {};


        if (
            MediaRecorder.isTypeSupported(
                "audio/webm;codecs=opus"
            )
        ) {

            options = {
                mimeType:
                    "audio/webm;codecs=opus"
            };

        }


        mediaRecorder =
            new MediaRecorder(
                stream,
                options
            );


        mediaRecorder.ondataavailable =
            function (event) {

                if (event.data.size > 0) {

                    audioChunks.push(
                        event.data
                    );

                }

            };


        mediaRecorder.onstop =
            async function () {

                stream
                    .getTracks()
                    .forEach(function (track) {

                        track.stop();

                    });


                const audioBlob =
                    new Blob(
                        audioChunks,
                        {
                            type:
                                "audio/webm"
                        }
                    );


                await sendAudio(
                    audioBlob
                );

            };


        mediaRecorder.start();


        micButton
            .classList
            .add("recording");


        statusElement.textContent =
            "Listening...";


    } catch (error) {

        console.error(error);

        errorElement.textContent =
            "Microphone permission is required.";

    }

}



/* ==========================================================
   STOP RECORDING
   ========================================================== */

function stopRecording() {

    if (
        mediaRecorder &&
        mediaRecorder.state === "recording"
    ) {

        mediaRecorder.stop();


        micButton
            .classList
            .remove("recording");


        statusElement.textContent =
            "Thinking...";

    }

}



/* ==========================================================
   SEND AUDIO
   ========================================================== */

async function sendAudio(
    audioBlob
) {

    try {

        const formData =
            new FormData();


        formData.append(
            "audio",
            audioBlob,
            "recording.webm"
        );


        const response =
            await fetch(
                "/voice",
                {
                    method: "POST",
                    body: formData
                }
            );


        const result =
            await response.json();


        if (!result.success) {

            throw new Error(
                result.error ||
                "Something went wrong."
            );

        }


        transcriptElement.textContent =
            result.transcript;


        replyElement.textContent =
            result.reply;


        statusElement.textContent =
            "Ready!";


        speakResponse(
            result.speech_text,
            result.language
        );


    } catch (error) {

        console.error(error);


        statusElement.textContent =
            "Something went wrong.";


        errorElement.textContent =
            error.message;

    }

}



/* ==========================================================
   MOUSE
   ========================================================== */

micButton.addEventListener(
    "mousedown",
    startRecording
);


micButton.addEventListener(
    "mouseup",
    stopRecording
);


micButton.addEventListener(
    "mouseleave",
    stopRecording
);



/* ==========================================================
   TOUCH
   ========================================================== */

micButton.addEventListener(
    "touchstart",
    function (event) {

        event.preventDefault();

        startRecording();

    }
);


micButton.addEventListener(
    "touchend",
    function (event) {

        event.preventDefault();

        stopRecording();

    }
);

</script>

</body>

</html>
"""


# ============================================================
# HOME
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home():

    return HTML_PAGE


# ============================================================
# VOICE API
# ============================================================

@app.post("/voice")
async def voice(
    audio: UploadFile = File(...)
):

    temp_path = None

    try:

        # ----------------------------------------------------
        # READ AUDIO
        # ----------------------------------------------------

        audio_data = await audio.read()


        if not audio_data:

            return JSONResponse(
                {
                    "success": False,
                    "error":
                        "Audio is empty."
                },
                status_code=400
            )


        # ----------------------------------------------------
        # TEMP FILE
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".webm"
        ) as temp_file:

            temp_file.write(
                audio_data
            )

            temp_path = temp_file.name


        # ====================================================
        # GROQ WHISPER
        # ====================================================

        print(
            "\n[1] Sending audio to Groq Whisper..."
        )


        with open(
            temp_path,
            "rb"
        ) as audio_file:

            transcription = (
                client
                .audio
                .transcriptions
                .create(
                    file=audio_file,
                    model=STT_MODEL,
                    response_format="json",
                    temperature=0
                )
            )


        transcript = (
            transcription.text.strip()
        )


        print(
            "[Whisper]:",
            transcript
        )


        if not transcript:

            return JSONResponse(
                {
                    "success": False,
                    "error":
                        "I could not understand the audio."
                },
                status_code=400
            )


        # ====================================================
        # GROQ QWEN
        # ====================================================

        print(
            "[2] Sending question to Qwen..."
        )


        completion = (
            client
            .chat
            .completions
            .create(
                model=LLM_MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },

                    {
                        "role": "user",
                        "content": transcript
                    }
                ],

                temperature=0.4,

                max_completion_tokens=300,

                reasoning_effort="none",

                response_format={
                    "type": "json_object"
                }
            )
        )


        raw_reply = (
            completion
            .choices[0]
            .message
            .content
            .strip()
        )


        print(
            "[Qwen]:",
            raw_reply
        )


        # ====================================================
        # JSON PARSING
        # ====================================================

        try:

            ai_data = json.loads(
                raw_reply
            )


        except json.JSONDecodeError:

            match = re.search(
                r"\{.*\}",
                raw_reply,
                re.DOTALL
            )


            if match:

                try:

                    ai_data = json.loads(
                        match.group(0)
                    )

                except json.JSONDecodeError:

                    ai_data = {
                        "reply":
                            raw_reply,

                        "speech_text":
                            clean_for_speech(
                                raw_reply
                            )
                    }

            else:

                ai_data = {
                    "reply":
                        raw_reply,

                    "speech_text":
                        clean_for_speech(
                            raw_reply
                        )
                }


        # ====================================================
        # EXTRACT
        # ====================================================

        reply = str(
            ai_data.get(
                "reply",
                ""
            )
        ).strip()


        speech_text = str(
            ai_data.get(
                "speech_text",
                ""
            )
        ).strip()


        # ====================================================
        # FALLBACK
        # ====================================================

        if not reply:

            reply = (
                "Hmm... naan inga irukken! 😊"
            )


        if not speech_text:

            speech_text = clean_for_speech(
                reply
            )


        # ====================================================
        # LANGUAGE
        # ====================================================

        if contains_tamil(
            speech_text
        ):

            language = "tamil"

        else:

            language = "english"


        speech_text = clean_for_speech(
            speech_text
        )


        # ====================================================
        # RESPONSE
        # ====================================================

        return JSONResponse(
            {
                "success": True,

                "transcript":
                    transcript,

                "reply":
                    reply,

                "speech_text":
                    speech_text,

                "language":
                    language
            }
        )


    except Exception as e:

        print(
            "\nERROR:",
            repr(e)
        )


        return JSONResponse(
            {
                "success": False,

                "error":
                    str(e)
            },
            status_code=500
        )


    finally:

        if temp_path:

            try:

                if os.path.exists(
                    temp_path
                ):

                    os.remove(
                        temp_path
                    )

            except Exception:

                pass


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )