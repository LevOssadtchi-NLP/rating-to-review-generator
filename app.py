import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Загружаем модель и токенизатор (кэшируются при первом запуске)
@st.cache_resource
def load_model():
    model = GPT2LMHeadModel.from_pretrained("levos06/gpt2-large-finetuned")
    tokenizer = GPT2Tokenizer.from_pretrained("levos06/gpt2-large-finetuned")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model()

# Функция генерации
def generate_review(rate, max_length=100, num_return_sequences=1):
    prompt = f"Rate: {rate}, Text:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output_sequences = model.generate(
        **inputs,
        max_length=max_length + len(inputs["input_ids"][0]),
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id
    )

    reviews = []
    for seq in output_sequences:
        text = tokenizer.decode(seq, skip_special_tokens=True)
        review = text.split("Text:")[-1].strip()
        reviews.append(review)
    return reviews

# Интерфейс Streamlit
st.title("Генерация отзывов по рейтингу")
st.write("Выберите рейтинг и получите сгенерированные отзывы")

# Ползунок для рейтинга
rating_input = st.slider("Рейтинг", min_value=1, max_value=8, value=5, step=1)

# Количество вариантов отзыва
num_reviews = st.number_input("Количество отзывов", min_value=1, max_value=5, value=3, step=1)

# Кнопка запуска генерации
if st.button("Сгенерировать отзыв(ы)"):
    with st.spinner("Генерация отзывов..."):
        reviews = generate_review(rating_input, max_length=100, num_return_sequences=num_reviews)

    st.success("Готово! Вот ваши отзывы:")
    for i, r in enumerate(reviews):
        st.markdown(f"**Отзыв {i+1}:** {r}")

