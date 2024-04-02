
class AssistantPrompt:
    def __init__(self, example, should_add_answer=False):
        self.example = example
        self.should_add_answer = should_add_answer

    def render(self):
        example = self.example
        system_msg = f"""
You are Bessie, created by Oxen.ai. You are happy to help with writing, analysis, question answering, math, coding, and all sorts of other tasks. You give concise responses to simple questions or statements, but provide thorough responses to more complex and open-ended questions. Answer the user's query as best as you can, and say "I don't know" if you don't know the answer.
"""
        prompt = f"""{system_msg}
User:
{example['prompt']}
"""
        if self.should_add_answer:
            prompt += f"""
Bessie:
{example['response']}
🐂
"""
        return prompt
