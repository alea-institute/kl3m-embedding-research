"""
Example tokenization output to test kl3m-001-32k and kl3m-003-64k.
"""

# imports

# packages
from tokenizers import Tokenizer

if __name__ == "__main__":
    SAMPLE_TEXT = """The Comptroller of the Currency shall have the same authority with respect to functions transferred to
 the Comptroller of the Currency under the Enhancing Financial Institution Safety and Soundness Act of 2010 as was
 vested in the Director of the Office of Thrift Supervision on the transfer date, as defined in section 311 of that
 Act [12 U.S.C. 5411]."""

    # kl3m-001-32k
    tokenizer_001 = Tokenizer.from_pretrained(
        "alea-institute/kl3m-001-32k"
    )
    tokens_001 = tokenizer_001.encode(SAMPLE_TEXT)

    # kl3m-003-64k
    tokenizer_003 = Tokenizer.from_pretrained(
        "alea-institute/kl3m-003-64k"
    )
    tokens_003 = tokenizer_003.encode(SAMPLE_TEXT)

    # output original text and each token mapped
    print("Original text: ", SAMPLE_TEXT)

    # output size, ids, and tokens for each model
    print("\nkl3m-001-32k\n" + "-" * 20)
    print("Size: ", len(tokens_001.ids))
    print("Tokens: ", tokens_001.tokens)
    print("IDs: ", tokens_001.ids)

    print("\nkl3m-003-64k\n" + "-" * 20)
    print("Size: ", len(tokens_003.ids))
    print("Tokens: ", tokens_003.tokens)
    print("IDs: ", tokens_003.ids)
