# 학습 질문이 평가 세트(개발·형식·잡음·단문·공개·봉인)와 겹치는지 검사한다.
# 방법: 글자 2~4그램 TF-IDF 코사인 유사도 + 숫자 묶음 일치.
# 사용: python overlap_check.py 학습질문.jsonl 평가1.csv|jsonl 평가2 ... [--out 결과.csv]
import sys, json, csv, re, argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load(path):
    rows = []
    if path.endswith('.csv'):
        for r in csv.DictReader(open(path, encoding='utf-8-sig')):
            rows.append((r.get('문항번호') or r.get('id'), r.get('질문') or r.get('question')))
    else:
        for l in open(path, encoding='utf-8'):
            r = json.loads(l)
            rows.append((r.get('문항번호') or r.get('id') or r.get('qid'), r.get('질문') or r.get('question')))
    return [(i, q) for i, q in rows if q]

def nums(q):
    # 판정에 쓰이는 숫자(금액·날짜·비율)를 모아 비교한다
    return set(re.findall(r'\d+(?:[.,]\d+)?', q.replace('，', ',')))

ap = argparse.ArgumentParser()
ap.add_argument('train'); ap.add_argument('evals', nargs='+')
ap.add_argument('--th', type=float, default=0.55, help='이 값 이상이면 겹침 의심')
ap.add_argument('--out', default='겹침검사_결과.csv')
a = ap.parse_args()

T = load(a.train)
E = []
for p in a.evals:
    E += load(p)
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), sublinear_tf=True)
vec.fit([q for _, q in T] + [q for _, q in E])
S = cosine_similarity(vec.transform([q for _, q in T]), vec.transform([q for _, q in E]))

out = []
for i, (tid, tq) in enumerate(T):
    j = S[i].argmax(); s = S[i, j]
    shared = nums(tq) & nums(E[j][1])
    flag = s >= a.th or (s >= 0.4 and len(shared) >= 2)
    out.append({'학습문항': tid, '최고유사도': round(float(s), 3), '가장비슷한평가문항': E[j][0],
                '겹친숫자': ' '.join(sorted(shared)), '의심': '예' if flag else '', '학습질문': tq, '평가질문': E[j][1]})
with open(a.out, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
sus = [o for o in out if o['의심']]
print(f'학습 {len(T)}개 / 평가 {len(E)}개 / 의심 {len(sus)}개 → {a.out}')
for o in sorted(sus, key=lambda o: -o['최고유사도'])[:40]:
    print(o['학습문항'], o['최고유사도'], o['가장비슷한평가문항'], o['겹친숫자'], '|', o['학습질문'][:50], '||', o['평가질문'][:50])
