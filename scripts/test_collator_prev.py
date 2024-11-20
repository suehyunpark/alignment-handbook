import torch
from transformers import AutoTokenizer
import pytest
from typing import List, Dict
from run_sft import DataCollatorForAssistantOnlyLM
class TestDataCollatorForHeaderBasedMasking:
    @pytest.fixture
    def tokenizer(self):
        return AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    
    @pytest.fixture
    def collator(self, tokenizer):
        return DataCollatorForAssistantOnlyLM(
            tokenizer=tokenizer,
            mlm=False,
            ignore_index=-100
        )
    
    def create_conversation(self, text: str, tokenizer) -> Dict[str, torch.Tensor]:
        """Helper to tokenize a conversation."""
        return tokenizer(text, return_tensors="pt")
    
    def get_unmasked_sections(self, batch, batch_idx=0):
        """Helper to extract unmasked sections from batch."""
        unmasked_sections = []
        current_section = []
        for i, (input_id, label) in enumerate(zip(batch["input_ids"][batch_idx], batch["labels"][batch_idx])):
            if label != -100:
                current_section.append(input_id.item())
            elif current_section:
                unmasked_sections.append(current_section)
                current_section = []
        
        if current_section:
            unmasked_sections.append(current_section)
        return unmasked_sections
    
    def find_substring_tokens(self, full_text: str, substring: str, tokenizer) -> List[int]:
        """Helper to find token indices for a substring."""
        full_tokens = tokenizer.encode(full_text)
        sub_tokens = tokenizer.encode(substring)
        
        for i in range(len(full_tokens) - len(sub_tokens) + 1):
            if full_tokens[i:i+len(sub_tokens)] == sub_tokens:
                return list(range(i, i+len(sub_tokens)))
        return []

    # Original tests remain unchanged
    def test_basic_masking(self, collator, tokenizer):
        """Test that only assistant responses are unmasked."""
        conversation = """<|start_header_id|>system<|end_header_id|>You are helpful
<|start_header_id|>user<|end_header_id|>Hi
<|start_header_id|>assistant<|end_header_id|>Hello!
<|start_header_id|>user<|end_header_id|>How are you?
<|start_header_id|>assistant<|end_header_id|>Great!"""
        
        inputs = self.create_conversation(conversation, tokenizer)
        batch = collator.torch_call([inputs["input_ids"][0].tolist()])
        
        unmasked_sections = self.get_unmasked_sections(batch)
        decoded = tokenizer.decode([t for section in unmasked_sections for t in section])
        
        assert "Hello!" in decoded
        assert "Great!" in decoded
        assert "You are helpful" not in decoded
        assert "Hi" not in decoded
    
    # ... [previous test methods remain unchanged] ...

    # New chat template specific tests
    def test_chat_template_masking(self, collator, tokenizer):
        """Test masking with complete chat template format."""
        conversation = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
2+2 equals 4.<|eot_id|><|start_header_id|>ipython<|end_header_id|>
print(2+2)<|eot_id|><|start_header_id|>assistant<|end_header_id|>
The output shows 4.<|eot_id|>"""
        
        inputs = self.create_conversation(conversation, tokenizer)
        batch = collator.torch_call([inputs["input_ids"][0].tolist()])
        
        unmasked_sections = self.get_unmasked_sections(batch)
        decoded_sections = [tokenizer.decode(section) for section in unmasked_sections]
        
        # Verify only assistant responses are unmasked
        assert len(decoded_sections) == 2
        assert any("2+2 equals 4" in section for section in decoded_sections)
        assert any("The output shows 4" in section for section in decoded_sections)
        
        # Verify other messages are masked
        labels = batch["labels"][0]
        input_text = tokenizer.decode(batch["input_ids"][0])
        
        for text, role in [
            ("You are a helpful assistant", "system"),
            ("What is 2+2?", "user"),
            ("print(2+2)", "ipython")
        ]:
            indices = self.find_substring_tokens(input_text, text, tokenizer)
            assert all(labels[i] == -100 for i in indices if i < len(labels)), f"{role} message not properly masked"

    def test_special_tokens_masking(self, collator, tokenizer):
        """Test masking of special tokens in chat template."""
        conversation = """<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>
Hello world!<|eot_id|>"""
        
        inputs = self.create_conversation(conversation, tokenizer)
        batch = collator.torch_call([inputs["input_ids"][0].tolist()])
        
        input_text = tokenizer.decode(batch["input_ids"][0])
        labels = batch["labels"][0]
        
        # Special tokens should be masked
        for token in ["<|begin_of_text|>", "<|eot_id|>"]:
            indices = self.find_substring_tokens(input_text, token, tokenizer)
            assert all(labels[i] == -100 for i in indices if i < len(labels)), f"{token} not properly masked"
        
        # Content should be unmasked
        hello_indices = self.find_substring_tokens(input_text, "Hello world", tokenizer)
        assert any(labels[i] != -100 for i in hello_indices if i < len(labels)), "Assistant content improperly masked"

    def test_mixed_content_chat_template(self, collator, tokenizer):
        """Test chat template with mixed content types."""
        conversation = """<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>
Here is some code:
```python
print("hello")
```
And some text.<|eot_id|>"""
        
        inputs = self.create_conversation(conversation, tokenizer)
        batch = collator.torch_call([inputs["input_ids"][0].tolist()])
        
        unmasked_sections = self.get_unmasked_sections(batch)
        decoded = tokenizer.decode([t for section in unmasked_sections for t in section])
        
        assert "Here is some code" in decoded
        assert "print" in decoded
        assert "And some text" in decoded
        
        # Special tokens should be masked
        labels = batch["labels"][0]
        input_text = tokenizer.decode(batch["input_ids"][0])
        for token in ["<|begin_of_text|>", "<|eot_id|>"]:
            indices = self.find_substring_tokens(input_text, token, tokenizer)
            assert all(labels[i] == -100 for i in indices if i < len(labels))

    def test_consecutive_chat_messages(self, collator, tokenizer):
        """Test consecutive messages in chat template format."""
        conversation = """<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>
First message<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Second message<|eot_id|>"""
        
        inputs = self.create_conversation(conversation, tokenizer)
        batch = collator.torch_call([inputs["input_ids"][0].tolist()])
        
        unmasked_sections = self.get_unmasked_sections(batch)
        decoded_sections = [tokenizer.decode(section) for section in unmasked_sections]
        
        assert len(decoded_sections) == 2
        assert any("First message" in section for section in decoded_sections)
        assert any("Second message" in section for section in decoded_sections)