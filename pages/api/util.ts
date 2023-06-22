import { OpenAIChat } from 'langchain/llms/openai';
import { LLMChain, ConversationalRetrievalQAChain, loadQAChain } from 'langchain/chains';
import { FaissStore } from 'langchain/vectorstores/faiss';
import { PromptTemplate } from 'langchain/prompts';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `Você é um assistente de IA do site Storyverse. O site está localizado em http://familyverse.bagagemlab.com/.
  Você recebe as como entrada partes extraídas do site e uma pergunta. Forneça uma resposta com um hiperlink (http://familyverse.bagagemlab.com/)
  para documentação somente quando necessário.
  Você só deve usar hiperlinks explicitamente listados como fonte no contexto. NÃO crie um hiperlink que não esteja listado.
  Quando solicitado a criar algo novo, use sua criatividade para criar algo que pareça natural e que não seja uma cópia exata do contexto.
  Se você não souber a resposta, apenas diga "Hmm, não tenho certeza". Não tente inventar uma resposta.
Question: {question}
=========
{context}
=========
Answer in Markdown:`
);

export const makeChain = (vectorstore: FaissStore, onTokenStream?: (token: string) => void) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ 
      temperature: 0,
      azureOpenAIApiKey: process.env.AZURE_OPENAI_API_KEY,
      azureOpenAIApiInstanceName: process.env.AZURE_OPENAI_API_INSTANCE_NAME,
      azureOpenAIApiDeploymentName: process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME,
      azureOpenAIApiVersion: process.env.AZURE_OPENAI_API_VERSION,
      maxTokens: 2048 }),
    prompt: CONDENSE_PROMPT,
  });

  const model = new OpenAIChat({
    temperature: 0.7,
    azureOpenAIApiKey: process.env.AZURE_OPENAI_API_KEY,
    azureOpenAIApiInstanceName: process.env.AZURE_OPENAI_API_INSTANCE_NAME,
    azureOpenAIApiDeploymentName: process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME,
    azureOpenAIApiVersion: process.env.AZURE_OPENAI_API_VERSION,
    maxTokens: 2048,
    streaming: Boolean(onTokenStream),
    callbacks: [
      {
        handleLLMNewToken: onTokenStream,
      },
    ],
  });

  const docChain = loadQAChain(model, { type: 'stuff', prompt: QA_PROMPT });

  return new ConversationalRetrievalQAChain({
    retriever: vectorstore.asRetriever(),
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
  });
};
