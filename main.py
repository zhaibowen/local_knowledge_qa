import os
import re
import time
import faiss
import numpy as np
import gradio as gr
from docx import Document
from functools import partial
from text2vec import SentenceModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_DICT = {
    "system": (
        "```\n"
        "{}"
        "```\n"
        "你是一个assistant, 上文是从《员工手册》中截取的一段话，请根据上文内容回答user的问题，你的回答要在300字以内\n"
    ),
    "query": (
        "\nuser: {}\n"
        "\nassistant: "
    )
}

def split_file_by_section(filepath):
    data = Document(filepath)
    lines = list(filter(lambda x: len(x.text.strip()) > 0, data.paragraphs))
    lines = list(map(lambda x: x.text.strip(), lines))

    body = lines[19:872]
    contents = []
    chapter_contents = []
    section_contents = []
    sentences = []
    sentence_map = {}
    for line in body:
        if re.match('^第.*章', line):
            chapter_contents.append(section_contents)
            contents.append(chapter_contents)
            chapter_contents = []
            section_contents = []
        elif re.match('^第.*节', line):
            chapter_contents.append(section_contents)
            section_contents = []
        else:
            ss = list(filter(lambda x: len(x) > 1, line.split("。")))
            for s in ss:
                sentence_map[len(sentences)] = [len(contents)-1, len(chapter_contents), len(section_contents)]
                sentences.append(s)
        section_contents.append(line)

    chapter_contents.append(section_contents)
    contents.append(chapter_contents)
    contents = contents[1:]

    section_length = []
    for chapter in contents:
        section_length.append([])
        for section in chapter:
            length = 0
            for line in section:
                length += len(line)
            section_length[-1].append(length)

    return contents, sentences, sentence_map, section_length

def encode_sentence_embedding(model, sentence_list):
    embeddings = model.encode(sentence_list)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

def get_prompt(chapter, sec, lid, section_lengths, max_length=2000):
    promt = '\n'.join(chapter[0]) + '\n'
    if sec == 0:
        limit_length = max_length // (len(chapter) - 1)
        for section in chapter[1:]:
            promt += '\n'.join(section)[:limit_length] + '\n'
    elif section_lengths[sec] < max_length:
        promt += '\n'.join(chapter[sec]) + '\n'
    else:
        clid = lid
        spromt = ""
        while len(spromt) < max_length / 2 and clid >= 0:
            spromt = chapter[sec][clid] + '\n' + spromt
            clid -= 1
        clid = lid + 1
        while len(spromt) < max_length and clid < len(chapter[sec]):
            spromt += chapter[sec][clid] + '\n'
            clid += 1
        promt += spromt

    return promt

def get_most_relavant_id_by_cluster(I, sentence_map):
    cluster_map = {}
    for i in I[0]:
        cha, sec, lid = sentence_map[i]
        if cha not in cluster_map:
            cluster_map[cha] = [0, []]
        cluster_map[cha][0] += 1
        cluster_map[cha][1].append(i)

    sorted_cluster = sorted(cluster_map.items(), key=lambda x: x[1][0], reverse=True)
    return sorted_cluster[0][1][1][0], cluster_map
        
def get_promt_with_query(model, index, query, section_list, sentence_list, sentence_map, section_length):
    query_embeddings = model.encode([query])
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    D, I = index.search(query_embeddings, 3)
    sentence_id, cluster_map = get_most_relavant_id_by_cluster(I, sentence_map)
    most_relavent_sentence = sentence_list[sentence_id]
    cha, sec, lid = sentence_map[sentence_id]
    promt = get_prompt(section_list[cha], sec, lid, section_length[cha])
    promt_system = PROMPT_DICT['system'].format(promt)
    promt_query = PROMPT_DICT['query'].format(query)
    return promt_system, promt_query, D, I, most_relavent_sentence, cha, sec, lid, cluster_map

def qwen_response(query, history, model, tokenizer, text2vec_model, index, section_list, sentence_list, sentence_map, section_length):
    begin_time = time.time()
    promt_system, promt_query, D, I, most_relavent_sentence, cha, sec, lid, cluster_map = \
            get_promt_with_query(text2vec_model, index, query, section_list, sentence_list, sentence_map, section_length)
    print(f"most_relavent_sentence:\n{most_relavent_sentence}\n\n\n")
    print(promt_system)
    print(promt_query)
    response, _ = model.chat(tokenizer, query, history=[], system=promt_system)
    print(response)
    print(f"query_length: {len(promt_system) + len(promt_query)}, response_length: {len(response)}, consume {time.time() - begin_time:.1f}s")
    return response

def main(llm_model_path, text2vec_model_path, file_path, index_path):
    text2vec_model = SentenceModel(text2vec_model_path, device='cuda')
    section_list, sentence_list, sentence_map, section_length = split_file_by_section(file_path)
    embedding_list = encode_sentence_embedding(text2vec_model, sentence_list)
    index = faiss.IndexFlatIP(768)
    index.add(embedding_list)
    faiss.write_index(index, index_path)

    # index = faiss.read_index(index_path)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(llm_model_path, device_map={'': 0}, trust_remote_code=True)
    model = model.eval()
    
    demo = gr.ChatInterface(partial(qwen_response, model=model, tokenizer=tokenizer, text2vec_model=text2vec_model,
                                    index = index, section_list=section_list, sentence_list=sentence_list, 
                                    sentence_map=sentence_map, section_length=section_length), title="Chat Bot")
    demo.launch(server_port=7080)

if __name__ == "__main__":
    llm_model_path = "/home/work/Qwen-7B-Chat-Int4"
    text2vec_model_path = '/home/work/disk/text2vec-base-chinese-paraphrase'
    file_path = "/home/work/disk/vision/program/n20231114_local_knowledge_based_qa/sample/员工手册.docx"
    index_path = '/home/work/disk/vision/program/n20231114_local_knowledge_based_qa/checkpoint/handbook.index'
    main(llm_model_path, text2vec_model_path, file_path, index_path)
