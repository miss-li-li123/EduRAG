from FlagEmbedding import BGEM3FlagModel

# 加载模型
model = BGEM3FlagModel('../rag_qa/models/bge-m3', use_fp16=False, device='cpu')

# 生成向量
text = "保险理赔需要准备哪些材料"

# 同时获取稠密向量和稀疏向量
output = model.encode(text,
                       return_dense=True,
                       return_sparse=True)

dense_vec = output['dense_vecs']  # 稠密向量 (1024,)
sparse_vec = output['lexical_weights']  # 稀疏向量 {词ID: 权重}

print(f"稠密向量维度: {len(dense_vec)}")
print(f"稀疏向量: {sparse_vec}")

# ===== 根据词id查询词 =====
# 获取tokenizer
tokenizer = model.tokenizer
# 反查token id对应的词
for token_id, weight in sorted(sparse_vec.items(), key=lambda x: x[1], reverse=True):
    token = tokenizer.decode([int(token_id)])
    print(f"  ID {token_id}: '{token}' → 权重 {weight:.4f}")