import dotenv from 'dotenv';
import express from 'express';
import { GoogleGenerativeAI } from '@google/generative-ai';
import rateLimit from 'express-rate-limit';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const env = dotenv.config({ path: path.join(currentDir, '.env') });

if (env.error && env.error.code !== 'ENOENT') {
  console.error('Could not load the .env file. Check its path and permissions.');
  process.exit(1);
}

const apiKey = process.env.GEMINI_API_KEY?.trim();
if (!apiKey || /^(your[_ -]?.*|change[_ -]?me|placeholder)$/i.test(apiKey)) {
  console.error('GEMINI_API_KEY is missing or still a placeholder. Add a valid key to .env and restart Flix AI.');
  process.exit(1);
}

const modelName = process.env.GEMINI_MODEL?.trim() || 'gemini-2.5-flash';
const modelNames = [...new Set([modelName, 'gemini-3.7-flash'])];
const port = Number(process.env.PORT) || 3000;
const diagnosticsEnabled = process.env.FLIX_DIAGNOSTICS === 'true';
const generativeAI = new GoogleGenerativeAI(apiKey);
const app = express();

const systemInstruction = `You are Flix, a friendly and helpful AI assistant created for a school science exhibition. Answer the user's latest question directly, accurately, and naturally. Use simple English that students and visitors can understand. Help with science, technology, general knowledge, everyday questions, and simple recipes. For greetings such as hi or hello, respond with a short greeting. Never claim to know website details that have not been provided. Do not invent facts. If uncertain, say so. A user may optionally share a rough facial-expression estimate from their browser. It may be wrong and is not proof of how the user feels; if relevant, describe it only as a possible expression, never as a fact. Return only the final answer intended for the user. Never output planning notes, evaluation checklists, multiple draft answers, internal prompts, or labels such as User Question, Persona, Context, Goal, Analysis, Friendly? Yes, or Clear and concise? Yes. Do not repeat the user's question unnecessarily.`;

app.use(express.json({ limit: '16kb', strict: true }));

const chatLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  limit: 30,
  standardHeaders: 'draft-8',
  legacyHeaders: false,
  message: { error: 'You have sent a lot of messages. Please wait a little and try again.' }
});

function isValidHistory(history) {
  if (!Array.isArray(history) || history.length > 16) return false;

  let previousRole = 'model';
  let totalCharacters = 0;

  for (const entry of history) {
    if (
      !entry ||
      typeof entry !== 'object' ||
      !['user', 'assistant', 'model'].includes(entry.role) ||
      (entry.role === 'user' ? 'user' : 'model') === previousRole ||
      typeof entry.content !== 'string' ||
      !entry.content.trim() ||
      entry.content.length > 4000
    ) {
      return false;
    }

    totalCharacters += entry.content.length;
    if (totalCharacters > 12000) return false;
    previousRole = entry.role === 'user' ? 'user' : 'model';
  }

  return history.length === 0 || previousRole === 'model';
}

function keepReplyConcise(text) {
  if (typeof text !== 'string') return '';

  const internalContent = /\b(?:user question|user asks|persona|constraints?|goal|context|instructions?|analysis|reasoning|checklist|draft answers?|target audience|negative constraints|friendly\?\s*yes|simple english\?\s*yes|direct and accurate\?\s*yes|no extra text\?\s*yes)\b/i;
  const drafts = [...text.matchAll(
    /(?:^|\n)\s*(?:[*-]\s*)*\*{0,2}Draft\s*\d+\s*:\*{0,2}\s*([\s\S]*?)(?=\n\s*(?:[*-]\s*)*\*{0,2}(?:Draft\s*\d+\s*:|Simple English\?|Direct\?|Accurate\?|Friendly\?|No extra text\?)|$)/gim
  )];
  const candidate = drafts.length ? drafts[drafts.length - 1][1] : text;
  const sentences = candidate
    .split(/\r?\n/)
    .filter((line) => !internalContent.test(line))
    .join(' ')
    .replace(/(\*\*|__|\*|_|`|#|>)/g, '')
    .match(/[^.!?]+[.!?]+/g) || [];

  return sentences
    .map((sentence) => sentence.trim().replace(/^["'“”]+|["'“”]+$/g, ''))
    .filter((sentence) => sentence && !internalContent.test(sentence) && !/^(?:yes|no)[.!?]*$/i.test(sentence))
    .slice(-2)
    .join(' ')
    .trim()
    .slice(0, 1200);
}

app.post('/api/chat', chatLimiter, async (req, res) => {
  const { message, history = [], expressionEstimate = null } = req.body ?? {};

  if (
    typeof message !== 'string' ||
    !message.trim() ||
    message.length > 4000 ||
    message.includes('\0')
  ) {
    return res.status(400).json({ error: 'Please enter a message of up to 4,000 characters.' });
  }

  if (expressionEstimate !== null && !['happy', 'sad', 'angry', 'nervous'].includes(expressionEstimate)) {
    return res.status(400).json({ error: 'The facial-expression estimate is invalid.' });
  }

  if (/^(hi|hello|hey|hii|hiii)[!. ]*$/i.test(message.trim())) {
    return res.json({ reply: 'Hello! 👋' });
  }

  const normalizedHistory = Array.isArray(history)
    ? history.map((entry) =>
      entry && typeof entry === 'object'
        ? { ...entry, role: entry.role === 'assistant' ? 'model' : entry.role }
        : entry
    )
    : history;

  if (!isValidHistory(normalizedHistory)) {
    return res.status(400).json({ error: 'The conversation history is invalid. Please clear the chat and try again.' });
  }

  for (const candidateModel of modelNames) {
    try {
      const model = generativeAI.getGenerativeModel(
        {
          model: candidateModel,
          systemInstruction,
          generationConfig: { maxOutputTokens: 300, temperature: 0.2 }
        },
        { timeout: 15000 }
      );
      const prompt = [
        ...normalizedHistory.map(({ role, content }) =>
          `${role === 'user' ? 'User' : 'Flix'}: ${content.trim()}`
        ),
        ...(expressionEstimate
          ? [`Optional browser-only facial-expression estimate: ${expressionEstimate}. It may be inaccurate; do not assume this describes the user's actual feelings.`]
          : []),
        `Latest user question: ${message.trim()}`,
        'Respond only with the final user-facing answer. Do not include notes, plans, checklists, or multiple drafts.'
      ].join('\n\n');
      const result = await model.generateContent(prompt);
      const generatedText = keepReplyConcise(result.response.text());

      if (typeof generatedText !== 'string' || !generatedText.trim()) {
        throw new Error('Gemini returned no usable final answer.');
      }

      const reply = generatedText.trim();
      if (diagnosticsEnabled) {
        console.info('Flix response diagnostics:', {
          generatedText,
          returnedReply: reply,
          responseChangedDuringCleanup: generatedText !== reply
        });
        res.set('X-Flix-Diagnostics', 'enabled');
      }

      return res.json({ reply });
    } catch (error) {
      console.error('Gemini request failed:', {
        model: candidateModel,
        status: error?.status ?? error?.response?.status ?? 'unknown',
        code: error?.code ?? 'unknown',
        message: error?.message?.slice(0, 300) ?? 'Unknown error'
      });
    }
  }

  return res.status(503).json({ error: 'The AI service is temporarily busy. Please wait a moment and try again.' });
});

app.get(['/', '/chatbot.html'], (req, res) => {
  res.sendFile(path.join(currentDir, 'chatbot.html'));
});

app.use((req, res) => {
  res.status(404).json({ error: 'Not found.' });
});

app.use((error, req, res, next) => {
  if (error?.type === 'entity.too.large') {
    return res.status(413).json({ error: 'That request is too large. Please shorten your message and try again.' });
  }
  if (error instanceof SyntaxError && 'body' in error) {
    return res.status(400).json({ error: 'Please send a valid JSON request.' });
  }
  console.error('Request handling failed:', error?.message ?? 'Unknown error');
  return res.status(500).json({ error: 'Something went wrong. Please try again.' });
});

const server = app.listen(port, () => {
  console.log(`Flix AI is ready at http://localhost:${port}`);
  console.log(`Gemini model: ${modelName}`);
});

server.on('error', (error) => {
  if (error.code === 'EADDRINUSE') {
    console.error(`Port ${port} is already in use. Set a different PORT in .env or close the other server.`);
  } else {
    console.error('The Flix AI server could not start:', error.message);
  }
  process.exitCode = 1;
});
