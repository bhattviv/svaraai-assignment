## 1. If you only had 200 labeled replies, how would you improve the model without collecting thousands more?

With only 200 labeled replies, I would use **data augmentation** techniques like back-translation, synonym replacement, or paraphrasing to increase training examples.

---

## 2. How would you ensure your reply classifier doesn’t produce biased or unsafe outputs in production?

I would first **analyze the training data** for class imbalance or biased language and apply preprocessing or reweighting to reduce bias. During deployment, I would implement **post-processing filters and monitoring** to catch unsafe, offensive, or irrelevant outputs.
---

## 3. Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?

I would provide **context-rich prompts**, including recipient information (name, role, company) and a specific goal for the email. I would also include **examples of desired outputs** in the prompt (few-shot learning) to guide the LLM. 
