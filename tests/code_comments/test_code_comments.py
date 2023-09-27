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
import lmql

from confirms.core.llm.gpt_lang_chain_llm import GptLangChainLlm

prompt = """Using this context ```["Min value is less than max value", "Max value is obtained externally"]```, 
comment the code below. All comments are on a separate line from code and should match the context exactly.
Do not try to comment the code with new text, instead find the most suitable part of the context. 
```
x = get_max_value()
y = x-1
```
"""


def test_code_commenting():
    """Test code commenting using provided description."""
    code_context = """
    def get_coupon_schedule(start_date, end_date, max_increase, spread):
        coupon_schedule = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        while current_date < end_date:
            current_end_date = current_date + relativedelta(months=3)
            libor_values = self.get_libor_value(current_date)
            if current_date < datetime(2009, 11, 13):
                coupon_rate = libor_values + spread
            else:
                coupon_rate = min(max(previous_coupon, libor_values + spread),previous_coupon + max_increase)
            coupon_amount = coupon_rate * (current_end_date - current_date).days / 365
            coupon_schedule.append((current_date.strftime('%Y-%m-%d'), coupon_amount))
            current_date = current_end_date
            previous_coupon = coupon_rate
        return coupon_schedule
    """
    document_context = """
    From (and including) 13 August 2009 to (but excluding) 13 November 2009 interest shall be payable quarterly 
    in arrears and accrue at a per annum rate determined according to the following formula: 
    3 month GBP Libor plus 0.40%
    From (and including) 13 November 2009 to (but excluding) 13 August 2012 interest shall be payable quarterly in 
    arrears and accrue at a per annum rate determined according to the following formula: 
    Min[ Max[ Previous Coupon, 3 month GBP Libor plus 0.40%], Previous Coupon +Max Increase]
    Where: 
    Max Increase is 0.25% 
    Previous coupon is the coupon paid on the previous Interest Payment Date. 
    3 month GBP Libor is the three-month GBP-LIBOR-BBA observed on Reuters Page LIBOR01 set two 
    London Banking Days prior to the commencement of the Interest Rate Calculation Period
    """
    initial_prompt = f"""
    You are provided with two types of content: CodeContext and DocumentContext.
    CodeContext consists of code blocks are separated by blank lines
    DocumentContext consists of sentences.
    Inside each function, insert a comment before each code block based on the sentences from the DocumentContext that is most relevant to this code block.
    Make sure each of the sentences in DocumentContext is included in output exactly once.
    Make sure Python indent rules are respected in comments places.
    Do not change the code lines in any way, only insert comments after blank lines and before the relevant code block.
    DocumentContext: ```{document_context}```
    CodeContext: ```{code_context}```
    """
    prompts = [initial_prompt, "No repetition", "Ok to rephrase a little", "Not as verbose"]
    model_types = ["gpt-3.5-turbo", "gpt-4"]
    for model_type in model_types:
        llm = GptLangChainLlm(model_type=model_type, temperature=0.0)
        print('_____')
        print(model_type)
        answers = llm.run_conversation_chain(prompts)
        print(*answers)
        print('_____')

