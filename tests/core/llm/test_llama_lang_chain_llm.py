# Copyright (C) 2023-present The Project Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pandas as pd
import pytest
from langchain import PromptTemplate

from confirms.core.llm.llama_lang_chain_llm import LlamaLangChainLlm


def test_smoke():
    """Run a smoke test to ensure each model is available."""

    llama_model_types = ["llama-2-7b-chat.Q4_K_M.gguf", "llama-2-13b-chat.Q4_K_M.gguf"]
    for model_type in llama_model_types:
        llm = LlamaLangChainLlm(model_type=model_type, temperature=0.0, seed=0)
        output = llm.completion("What is two times two?")
        assert output == "\n\nAnswer: Two times two is equal to 4."


def test_function_completion():
    """Function completion"""

    llama_model_types = [
        "llama-2-7b-chat.Q4_K_M.gguf"
    ]  # , "llama-2-13b-chat.Q4_K_M.gguf", "llama-2-70b-chat.Q4_K_M.gguf"]
    for model_type in llama_model_types:
        prompt = (
            "Act as a trade entry specialist whose goal is to extract parameters for a function "
            "generating interest rate schedule from the text specified by the user. "
            "These parameters are first_unadjusted_payment_date, last_unadjusted_payment_date, and"
            "payment_frequency."
            "Provide response only using the text specified by the user."
            "Do not make up answers. Use ISO 8601 format for dates, namely YYYY-MM-DD, in your answer."
        )
        question = (
            "First unadjusted payment date is on January 15, 2000, "
            "last unadjusted payment date is on January 15, 2005, and "
            "payment frequency is 6M."
        )
        llm = LlamaLangChainLlm(model_type=model_type, temperature=0.0, grammar_file="payment_schedule_params.gbnf")
        answer = llm.completion(question, prompt=prompt)
        pass
        # assert answer["first_unadjusted_payment_date"] == "2000-01-15"
        # assert answer["last_unadjusted_payment_date"] == "2005-01-15"
        # assert answer["payment_frequency"] == "6M"


def test_parameters_extraction():
    """Parameters extraction"""

    contexts = [
        (
            "Interest Payment Date: Interest payments shall be made quarterly on each 18th day of the months of "
            "April, July, October and January, commencing October 18, 2023."
        ),
        (
            "Settlement Date: 18-July-2023, Interest Payment Date: Interest payments shall be made quarterly on each "
            "18th day of the months of April, July, October and January, commencing October 18, 2023. Total number of "
            "payments is five."
        ),
        (
            "Issue Date: 9 July 2009 (Settlement Date), Maturity Date: 9 July 2013, Interest Payment Dates: The 9th "
            "of each January, April, July, and October commencing 9 October 2009 with a final payment on the "
            "Maturity Date."
        ),
        (
            "Issue Date: On or about December 27, 2013, Maturity Date and Term: On or about December 27, 2023, "
            "resulting in a term to maturity of approximately 10 years. The $100 principal amount (the 'Principal "
            "Amount') will only be payable at maturity. For further information, see 'Payments under the Notes'.,"
            "Interest Payment Date: The first Interest payment, if any, shall be made on June 27, 2014, following "
            "which Holders of the Notes will be entitled to receive semi-annual Interest payments, if any. "
            "Subject to the occurrence of certain Extraordinary Events, Interest, if any, will be payable on the "
            "27th day of June and December of each year that the Notes remain outstanding (each, an 'Interest Payment"
            " Date') from and including June 27, 2014 to and including the Maturity Date. If any Interest Payment "
            "Date is not a Business Day, it will be postponed to the next following Business Day"
        ),
    ]

    template = """
    <s>[INST] <<SYS>>Act as a trade entry specialist whose goal is to extract parameters for a function generating
    interest rate schedule from the context. These parameters are first_unadjusted_payment_date,
    last_unadjusted_payment_date, and payment_frequency.
    Provide response only using the text in context. Do not make up answers. Use ISO 8601 format for dates,
    namely YYYY-MM-DD, in your answer. \n <</SYS>> \n\n
    Context: {context}.
    Only return the helpful answer below and nothing else.
    Helpful answer:[/INST]
    """

    all_results = {}
    prompt = PromptTemplate(template=template, input_variables=["context"])
    llama_model_types = ["llama-2-7b-chat.Q4_K_M.gguf", "llama-2-13b-chat.Q4_K_M.gguf"]
    for model_type in llama_model_types:
        results = []
        for context in contexts:
            llm = LlamaLangChainLlm(model_type=model_type, temperature=0.0, grammar_file="payment_schedule_params.gbnf")
            answer_grammar = llm.completion(context, prompt=prompt).split(',')
            current_dict = {
                'context': context,
                'first_unadjusted_payment_date': answer_grammar[0].split('=')[1],
                'last_unadjusted_payment_date': answer_grammar[1].split('=')[1],
                'payment_frequency': answer_grammar[2].split('=')[1],
            }
            results.append(pd.DataFrame([current_dict]))
        all_results[model_type] = pd.concat(results)
    outputs_dir = os.path.join(os.path.dirname(__file__), "../../../outputs")
    output_path = os.path.join(outputs_dir, "test_parameters_extraction.csv")

    # TODO: Refactor saving of results
    df = pd.concat(all_results).reset_index()
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    pytest.main([__file__])
