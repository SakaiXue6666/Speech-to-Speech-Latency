已知src speech自带src transcript timestamp、tgt speech自带tgt reference  

1. 使用WhisperX / (Qwen3-ASR+Qwen3-ForcedAligner) 获得（1）tgt speech的tgt transcript和tgt timestamp  
2. 使用SEGALE做tgt hyp transcript和tgt ref transcript的分段 + 对齐  
3. 使用对齐关系 + tgt hyp timestamp计算latency  
