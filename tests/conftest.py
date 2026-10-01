import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HUMAN_TEXT = """Validation of the UK mortgage PD model (MDL-2231)

We re-ran the development sample (Jan 2015 to Dec 2019, 412,880 accounts) and reproduced the coefficients to four decimal places. The Gini on the out-of-time window (2021-01 to 2022-06) was 0.61 versus 0.66 in development; PSI of 0.08 is within appetite. Section 4.2 sets out the override analysis.

We disagree with the developer on the treatment of forbearance flags (see Finding 3). Accounts flagged under the 2020 payment holiday scheme were dropped from the sample, which removes about 7% of the 2020 cohort; that is not a small number. The calibration test (binomial, 95%) failed for grade 7 only. Ten of 12 grades passed.

Data lineage is poor: two fields (current LTV and bureau score at month 6) come from a spreadsheet maintained by one analyst in the Leeds office. We were told this will move to the warehouse in Q2 2024; we have heard that before. Until it does we'd treat the LTV input as a key-person dependency.

The developer's choice of a 12-month outcome window is fine. We tried 18 months and the rank ordering barely moved (Gini 0.60 vs 0.61). Our one real concern is the cut-off at grade 9, which was set by eye from Figure 3 rather than from any optimisation.
"""

AI_TEXT = """Conceptual Soundness Assessment

It is important to note that the model demonstrates robust performance across a comprehensive range of scenarios, ensuring alignment with regulatory expectations. Furthermore, the modelling framework fosters transparency and accountability, underscoring the pivotal role of governance in the model lifecycle.

Additionally, the methodology leverages a holistic approach to navigate the evolving regulatory landscape, thereby enhancing informed decision-making processes. Moreover, the selected variables are intuitive and well-aligned with industry best practices, providing valuable insights into key risk drivers.

Overall, the validation team concludes that the model is conceptually sound and fit for purpose. Notably, the comprehensive documentation serves as a testament to the meticulous development process, highlighting the commitment of key stakeholders to continuous improvement.

In conclusion, the model plays a crucial role in the bank's risk management framework, fostering a robust and seamless approach to credit risk assessment while ensuring the integrity and reliability of model outputs.

Ultimately, this comprehensive validation underscores the importance of a proactive approach to model risk, empowering stakeholders to strike a balance between innovation and prudence.
"""


@pytest.fixture
def human_text():
    return HUMAN_TEXT


@pytest.fixture
def ai_text():
    return AI_TEXT
