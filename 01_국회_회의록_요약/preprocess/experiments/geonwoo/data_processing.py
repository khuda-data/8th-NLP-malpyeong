import json
import hanja_module.hanjahangul as hanja
try:
    from src.tagging import apply_tags
except ModuleNotFoundError:
    import os, sys
    # 현재 파일 기준으로 상위 두 단계(preprocess) 경로를 PYTHONPATH에 추가
    ROOT_PREPROCESS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if ROOT_PREPROCESS not in sys.path:
        sys.path.append(ROOT_PREPROCESS)
    from src.tagging import apply_tags

INPUT_FILE = "국회회의록안건별요약_dev"
input_path = f"./data/{INPUT_FILE}.json"
# 고정 크기 청크 분할 길이 (예: 3이면 [0,2], [3,5], ...)
CHUNK_LEN = 10
CONFIG_PROMPT = """{PROMPT}

위 규칙에 따라 하나의 {step} 요약문을 생성하시오.

요약 중심 키워드: {keyword}
발언자:
{speaker}

이전 요약:
{previous_summary}

현재 대화:
{dialogue}
"""
# 주제: {topic}
CONFIG_SPEAKER = "<역할>{occupation}</역할> <이름>{name}</이름>"

def get_index_from_id(id_):
    return int(id_.split(".")[-1]) - 1

def preprocess_conversation(input_conversation, speaker_same_as_previous=False):
    return_text = (" " if speaker_same_as_previous else input_conversation["speaker"] + ": ") + input_conversation["utterance"]
    return_text = return_text.replace("......", " ") # <REMOVE_DOTS>

    return return_text

def speaker_to_text(speakers):
    return_text = "\n".join(
        list(
            map(
                lambda x: 
                    CONFIG_SPEAKER.replace("{occupation}", x['occupation']).replace("{name}", hanja.hanja_to_hangul_dueum(x['id'])),
                speakers
            )
        )
    )
    
    return return_text

def make_chunk_index_list_by_lengths(lengths, max_chunk_len):
    """길이 리스트를 기준으로 누적 길이가 max_chunk_len을 넘지 않도록 인덱스들을 묶는다.

    Args:
        lengths (List[int]): 각 원소의 길이 리스트.
        max_chunk_len (int): 청크의 최대 누적 길이.

    Returns:
        List[List[int]]: 인덱스 묶음 리스트. 예: [[0, 1], [2, 3], ...]
    """
    chunks = []
    current = []
    current_len = 0

    for i, l in enumerate(lengths):
        # 현재 청크에 추가 시 초과하는 경우, 청크를 마감하고 새로 시작
        if current and current_len + l > max_chunk_len:
            chunks.append(current)
            current = [i]
            current_len = l
        else:
            # 비어있을 때거나 초과하지 않을 때는 현재 청크에 추가
            current.append(i)
            current_len += l

        # 단일 원소가 max_chunk_len보다 긴 경우에도 단독 청크로 둔다
        if not current[:-1] and l > max_chunk_len:
            # 방금 추가한 단일 원소 청크를 바로 확정
            chunks.append(current)
            current = []
            current_len = 0

    if current:
        chunks.append(current)

    return chunks

def make_chunk_index_list(texts, max_chunk_len):
    """문자열 리스트를 받아 각 원소의 길이(len) 기준으로 청크 인덱스 리스트를 만든다.

    Args:
        texts (List[str]): 청크 대상으로 묶을 문자열 리스트.
        max_chunk_len (int): 청크의 최대 누적 길이.

    Returns:
        List[List[int]]: 인덱스 묶음 리스트. 예: [[0, 1], [2]]

    예시:
        texts 길이가 [10, 5, 6], max_chunk_len=20이면 -> [[0, 1], [2]]
    """
    lengths = [len(t) for t in texts]
    return make_chunk_index_list_by_lengths(lengths, max_chunk_len)

def make_fixed_size_chunk_ranges(n_items, chunk_len):
    """항목 개수와 고정 청크 크기를 받아 (start, end) 형태의 인덱스 범위 리스트를 반환한다.

    예: n_items=9, chunk_len=3 -> [(0,2), (3,5), (6,8)]
        n_items=10, chunk_len=3 -> [(0,2), (3,5), (6,8), (9,9)]
    """
    ranges = []
    if chunk_len <= 0:
        return ranges
    start = 0
    while start < n_items:
        end = min(start + chunk_len - 1, n_items - 1)
        ranges.append((start, end))
        start = end + 1
    return ranges

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

outputs = []

for sample in data:
    sample_inp = sample["input"]
    issue = sample_inp["issue"]

    speaker = sample_inp["speaker"]
    topic = "정의되지 않음" if len(issue["topic"]) == 0 else hanja.hanja_to_hangul_dueum(issue["topic"])
    keyword = issue['keyword']

    keyword_sentence_id = issue['sentence_id']
    keyword_sentence_int = get_index_from_id(keyword_sentence_id)

    begin_id = keyword_sentence_id
    end_id = sample_inp["conversation"][-1]["id"]

    is_end_found = False

    ids = []

    ########################################################
    # 프롬프트 설계
    ########################################################

    PROMPT = CONFIG_PROMPT.replace("{topic}", topic).replace("{keyword}", keyword).replace("{speaker}", speaker_to_text(speaker))

    ########################################################
    # 전처리 1단계
    # 
    # "선포"로 나누는 과정
    ########################################################

    for conv in sample["input"]["conversation"]:
        if "선포" in conv["utterance"]:
            id_ = conv["id"]
            id_int = get_index_from_id(id_)

            ids.append(id_)

            if not is_end_found and id_int > keyword_sentence_int:
                end_id = id_
                is_end_found = True

    ########################################################
    # 전처리 2단계
    #
    # begin_id ~ end_id 범위의 발화를 순회하며 필터링한다.
    #
    # 1) 1차 길이 필터: 공백과 '.'을 제거한 문자열 길이가 11자 이상인지 확인한다.
    #    - 11자 미만이면 즉시 제외.
    # 2) 길이 통과 후 포함 규칙:
    #    - 숫자 포함 문장: 포함.
    #    - '.......' 포함 문장: 정제 길이(공백·마침표 제거) >= 20 일 때만 포함.
    #    - 숫자도 '.......'도 없는 문장: 포함.
    #
    # 최종적으로 제외되는 경우는 두 가지:
    #    A. 정제 길이 <= 10
    #    B. 정제 길이 11~19 이면서 '.......' 포함
    #
    # 요약: 짧은 문장을 걸러내고(정제 길이 ≤10), 점선(".......")으로만 이루어지거나 망설임 성격의 짧은 문장은 더 엄격히 배제한다(정제 길이 11~19).
    ########################################################

    preprocessing_array = []

    for index in range(get_index_from_id(begin_id), get_index_from_id(end_id) + 1):
        conv = sample_inp["conversation"][index]
        utterance = conv["utterance"]

        speaker_same_as_previous = False

        if len(preprocessing_array) > 0:
            previous_conv_speaker = preprocessing_array[-1].split(": ")[0]
            if conv["speaker"] == previous_conv_speaker:
                speaker_same_as_previous = True

        # 필터 규칙: 정제 길이 > 10을 통과한 뒤
        #  - 숫자가 있으면 포함
        #  - '.......'이 있으면 정제 길이 >= 20일 때만 포함
        #  - 둘 다 없으면 포함

        # print(speaker_same_as_previous, len(preprocessing_array))

        filter_utterance = "".join("".join(utterance.split(" ")).split("."))
        speaker_and_utterance = (" " if speaker_same_as_previous else conv["speaker"] + ": ") + conv["utterance"]
        speaker_and_utterance = speaker_and_utterance.replace("......", ".")

        if len(filter_utterance) > 10:
            has_digit = any(char.isdigit() for char in utterance)
            has_dots = "......." in utterance

            if has_digit:
                if speaker_same_as_previous:
                    preprocessing_array[-1] += speaker_and_utterance
                else:
                    preprocessing_array.append(speaker_and_utterance)
            elif not has_digit:
                if has_dots:
                    if len(filter_utterance) >= 20:
                        if speaker_same_as_previous:
                            preprocessing_array[-1] += speaker_and_utterance
                        else:
                            preprocessing_array.append(speaker_and_utterance)
                else:
                    if speaker_same_as_previous:
                        preprocessing_array[-1] += speaker_and_utterance
                    else:
                        preprocessing_array.append(speaker_and_utterance)

    # print("keyword_sentence_id:", keyword_sentence_id)
    # print("begin_id:", begin_id)
    # print("end_id:", end_id)

    ###############################################
    # 비-분할 입력 생성
    ###############################################

    # FINAL_INPUT = PROMPT + "\n".join(preprocessing_array)
    # print(FINAL_INPUT)

    ###################################################
    # 분할 입력 생성
    ###################################################

    FINAL_INPUT = []

    for start_idx, end_idx in make_fixed_size_chunk_ranges(len(preprocessing_array), CHUNK_LEN):
        chunk_lines = preprocessing_array[start_idx:end_idx + 1]
        agenda_title = keyword if keyword else topic

        # 각 줄의 발화 부분에 태그 적용 (형식: "화자: 발화")
        tagged_lines = []
        for line in chunk_lines:
            if ": " in line:
                prefix, utter = line.split(": ", 1)
                tagged = apply_tags(utter, agenda_title=agenda_title, role=None)
                tagged_lines.append(f"{prefix}: {tagged}")
            else:
                tagged_lines.append(apply_tags(line, agenda_title=agenda_title, role=None))

        tagged_chunk_text = "\n".join(tagged_lines)

        # print("----- CHUNK START (TAGGED) -----")
        # print(tagged_chunk_text)
        # print("----- CHUNK END -----\n\n")

        FINAL_INPUT.append(
            PROMPT
                .replace("{dialogue}", tagged_chunk_text)
        )

    outputs.append({
        "id": sample["id"],
        "inputs": FINAL_INPUT,
        "outputs": sample["output"]
    })

with open(f"./experiments/geonwoo/processed/processed_{INPUT_FILE}.json", "w", encoding="utf-8") as f:
    json.dump(outputs, f, ensure_ascii=False, indent=2)