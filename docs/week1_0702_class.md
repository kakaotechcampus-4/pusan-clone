### 공통 system prompt 관련 코드 예시

```python
CHAT_MEMORY_PROMPT = (
    "너는 kana agent로 개인일정을 관리하는 schedule assistant야."
    "personal_create_schedule을 사용해서 사용자의 요청에 따라 일정을 등록해"
    "사용자가 삭제를 요청하면 아래 지시사항에 따라 처리해"
    "1. personal list schedules을 활용하여 일정을 조회한다"
    "2. 사용자가 삭제 요청한 일정의 아이디를 검색한다"
    "3. peronal delete schedule을 이용하여 일정을 삭제한다"
)
```

### few shot과 CoT prompt를 아래 함수에 어떻게 적용해야할지
- agent를 구성할 때 : few shot과 CoT prompt
- PE(prompt engineering) : zero shot, few shot, instruction, CoT
- 학습(Fine tuning) X : zero shot, few shot
- (zero shot)만약 내 의도대로 동작하지 않는다면... 그때 예시를 추가해도 됨(few shot)
- few shot(예시)과 CoT(chain of tool: 동작 순서)를 많이 사용하는 이유
  - 가정 : 질문) 45인승 버스에 골프공을 넣으면 몇개가 들어갈까?
  - CoT = workflow
  - agent를 만든다는 것 : 특정 분야의 문제를 잘 해결하는 것


### Agent system
agent의 동작을 기억
1. 일정을 등록할 때는 personal create schedule 라는 일정 확인 tool을 사용해서 등록한다
2. 일정을 조회할때는 list ~
3. 일정을 삭제할 때는 delete~
    - 보통 삭제를 할 때는 list schedule을 이용하여 전체 조회 후 Id 파악후 delete를 진행함

- tool을 trace하고 점검하다 보면 내가 생각한대로 안됨 -> 최대한 제한을 tool쪽으로 끌고와서 개선함 
- (그냥 클로드한테 개선해줘! 하면 ruled base 대로 기존의 방식대로 새로운 함수를 추가하는 등으로 개선하려고 함... 제대로 개선 X)
- 따라서 tool 에 동작을 추가하며 내가 의도한 대로 tool을 끌고가야 됨
- 어떤 동작에 어떤식으로 진행해야 하는지 (claude favle 5 system prompt 문서 참고해서 자신의 agent에 적용해보기)

### `def personal_create_schedule`

- 구조가 명시되지 않았다면 few shot 등으로 json 구조를 만들 수 있음

```bash
사용자: 내일 7/3 민수랑 개발 미팅을 점심에 잡아줘
{
    title: 민수와 개발 미팅
    date: 2026-07-03
    start_time: 12
    end_time: 14
    ...
}
```