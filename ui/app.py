import streamlit as st
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

hf_token = os.getenv("HF_TOKEN")

@st.cache_resource
def load_model_and_tokenizer():
    model_name = "smartcat/T5-product2query-finetune-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=hf_token)
    return tokenizer, model


def generate_response(input_text, tokenizer, model, max_length=100):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    generated_ids = model.generate(
        inputs["input_ids"], max_length=30, num_beams=4, early_stopping=True
    )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text


def main():
    st.set_page_config(page_title="Query Generation", page_icon="🌐", layout="centered")

    st.title(
        """Query Generation with smartcat/T5-product2query-finetune-v1 :thinking_face:"""
    )
    st.write("Enter text below and press the button to get a generated response.")

    input_text = st.text_area(
        "Input Text:", placeholder="Type something...", height=250
    )

    if st.button("Generate Response"):
        if input_text.strip():
            with st.spinner("Generating response..."):
                tokenizer, model = load_model_and_tokenizer()
                generated_text = generate_response(input_text, tokenizer, model)
                st.subheader("Generated Response:")
                st.markdown(f""":gray-background[{generated_text}]""")
        else:
            st.warning("Please enter some text to generate a response.")

    option = st.selectbox(
    "Select an example", ["None", "Wateproof boots", "Bamboo socks", "Levi's Jeans"]
)

    if option == "Wateproof boots":
        st.subheader("Selected example text:")
        st.markdown("""Waterproof boot with synthetic upper and interior bootie, multi strap shaft detail with buckle closure, faux fur lined shaft and footbed, roxy metal logo pin, and flexible tpr unit bottom.
Roxy is a brand of Quiksilver, Inc., the world's leading outdoor sports lifestyle company, which designs, produces and distributes a diversified mix of branded apparel, footwear, accessories and related products. Quiksivler's apparel and footwear brands represent a casual lifestyle for young-minded people that connect with its boardriding culture and heritage, while outdoor sports symbolize a long standing commitment to technical expertise and competitive success. The company's products are sold in more than 90 countries.""")
        with st.spinner("Generating response..."):
            tokenizer, model = load_model_and_tokenizer()
            input_text = """Waterproof boot with synthetic upper and interior bootie, multi strap shaft detail with buckle closure, faux fur lined shaft and footbed, roxy metal logo pin, and flexible tpr unit bottom.
Roxy is a brand of Quiksilver, Inc., the world's leading outdoor sports lifestyle company, which designs, produces and distributes a diversified mix of branded apparel, footwear, accessories and related products. Quiksivler's apparel and footwear brands represent a casual lifestyle for young-minded people that connect with its boardriding culture and heritage, while outdoor sports symbolize a long standing commitment to technical expertise and competitive success. The company's products are sold in more than 90 countries.

"""
            generated_text = generate_response(input_text, tokenizer, model)
            st.subheader("Generated Response:")
            st.markdown(f""":gray-background[{generated_text}]""")

    elif option == "Bamboo socks":
        st.subheader("Selected example text:")
        st.markdown(
            """FINE THREADS: These socks are the ones you can ususally find in a high end store with elegance. And here are how to provide care:  Machine washable (cold).  Tumble dry low, absolutely no bleach. If possible dry naturally instead of using the dryer. Highly recommended to wash inside out.  Cotton and bamboo tends to shrink in machine wash especially when washing temperature rises. We recommend cold wash with delicate cycle."""
        )
        with st.spinner("Generating response..."):
            tokenizer, model = load_model_and_tokenizer()
            input_text = """FINE THREADS: These socks are the ones you can ususally find in a high end store with elegance. And here are how to provide care:  Machine washable (cold).  Tumble dry low, absolutely no bleach. If possible dry naturally instead of using the dryer. Highly recommended to wash inside out.  Cotton and bamboo tends to shrink in machine wash especially when washing temperature rises. We recommend cold wash with delicate cycle."""
            generated_text = generate_response(input_text, tokenizer, model)
            st.subheader("Generated Response:")
            st.markdown(f""":gray-background[{generated_text}]""")

    elif option == "Levi's Jeans":
        st.subheader("Selected example text:")
        st.markdown(
            """More than 140 years after inventing the blue jean, one thing is clear: Levi's clothes are loved by the people who wear them - from presidents to movie stars, farmers to fashion icons, entrepreneurs to the everyman. 'Live in Levi's' asserts with confidence and pride that Levi's clothes are indeed for everybody who's not just anybody."""
        )
        with st.spinner("Generating response..."):
            tokenizer, model = load_model_and_tokenizer()
            input_text = """More than 140 years after inventing the blue jean, one thing is clear: Levi's clothes are loved by the people who wear them - from presidents to movie stars, farmers to fashion icons, entrepreneurs to the everyman. 'Live in Levi's' asserts with confidence and pride that Levi's clothes are indeed for everybody who's not just anybody."""
            generated_text = generate_response(input_text, tokenizer, model)
            st.subheader("Generated Response:")
            st.markdown(f""":gray-background[{generated_text}]""")


if __name__ == "__main__":
    main()
