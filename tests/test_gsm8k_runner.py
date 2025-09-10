import budgetbench.runner as runner

def test_run_gsm8k_task(monkeypatch):
    dataset = [{
        "question": "What is 2 + 2?",
        "answer": "Add the numbers to get 4. #### 4",
    }]

    def fake_chat(prompt, model, max_tokens):
        assert prompt == "What is 2 + 2?"
        return {"message": "The answer is 4", "cost": {"total": 0.0}}

    monkeypatch.setattr(runner, "chat_completion", fake_chat)
    result = runner.run_gsm8k_task(0, model="test-model", dataset=dataset)
    assert result["passed"] == 1
    assert result["answer"] == "4"
    assert result["total"] == 1


def test_run_gsm8k_task_incorrect(monkeypatch):
    dataset = [{
        "question": "What is 2 + 2?",
        "answer": "Add the numbers to get 4. #### 4",
    }]

    def fake_chat(prompt, model, max_tokens):
        return {"message": "5", "cost": {"total": 0.0}}

    monkeypatch.setattr(runner, "chat_completion", fake_chat)
    result = runner.run_gsm8k_task(0, model="test-model", dataset=dataset)
    assert result["passed"] == 0
