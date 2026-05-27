"""Definitions of 12 cognitive biases with control/treatment prompt variants.

Each bias is a dict with:
  - name: str
  - category: str
  - bias_type: "numerical" | "categorical"
  - variants: list of dicts, each with:
      - id: str (e.g., "anchoring_v1")
      - control: str (prompt without bias manipulation)
      - treatment: str (prompt with bias manipulation)
      - parse_type: "number" | "choice"
      - anchor (optional): the anchor value for numerical biases
      - bias_option (optional): the option predicted by the bias for categorical biases
"""

# ============================================================
# Category 1: JUDGMENT UNDER UNCERTAINTY
# ============================================================

ANCHORING = {
    "name": "anchoring",
    "category": "judgment_under_uncertainty",
    "bias_type": "numerical",
    "variants": [
        {
            "id": "anchoring_v1",
            "control": (
                "The population of Chicago is a well-known fact. "
                "What is your best estimate of the population of Chicago? "
                "Provide only a number."
            ),
            "treatment": (
                "The population of Chicago is a well-known fact. "
                "A recent (incorrect) article claimed the population is 15 million. "
                "What is your best estimate of the population of Chicago? "
                "Provide only a number."
            ),
            "parse_type": "number",
            "anchor": 15_000_000,
        },
        {
            "id": "anchoring_v2",
            "control": (
                "Consider the average annual temperature in London. "
                "What is your best estimate in degrees Celsius? "
                "Provide only a number."
            ),
            "treatment": (
                "Consider the average annual temperature in London. "
                "Someone suggested it might be around 25°C. "
                "What is your best estimate in degrees Celsius? "
                "Provide only a number."
            ),
            "parse_type": "number",
            "anchor": 25,
        },
        {
            "id": "anchoring_v3",
            "control": (
                "How many countries are there in Africa? "
                "Provide only a number."
            ),
            "treatment": (
                "Do you think there are more or fewer than 15 countries in Africa? "
                "How many countries are there in Africa? "
                "Provide only a number."
            ),
            "parse_type": "number",
            "anchor": 15,
        },
        {
            "id": "anchoring_v4",
            "control": (
                "What is the height of the Eiffel Tower in meters? "
                "Provide only a number."
            ),
            "treatment": (
                "A tourist guessed the Eiffel Tower is about 500 meters tall. "
                "What is the height of the Eiffel Tower in meters? "
                "Provide only a number."
            ),
            "parse_type": "number",
            "anchor": 500,
        },
        {
            "id": "anchoring_v5",
            "control": (
                "What percentage of the Earth's surface is covered by water? "
                "Provide only a number."
            ),
            "treatment": (
                "An old textbook states that approximately 50% of the Earth's surface "
                "is covered by water. "
                "What percentage of the Earth's surface is covered by water? "
                "Provide only a number."
            ),
            "parse_type": "number",
            "anchor": 50,
        },
    ],
}

AVAILABILITY = {
    "name": "availability_heuristic",
    "category": "judgment_under_uncertainty",
    "bias_type": "categorical",
    "variants": [
        {
            "id": "availability_v1",
            "control": (
                "Which is a more common cause of death in the United States: "
                "(A) heart disease or (B) homicide? "
                "Answer with only A or B."
            ),
            "treatment": (
                "Recently, there have been many news reports about violent crimes "
                "and shootings across the United States. "
                "Which is a more common cause of death in the United States: "
                "(A) heart disease or (B) homicide? "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "B",
        },
        {
            "id": "availability_v2",
            "control": (
                "Which causes more deaths per year worldwide: "
                "(A) shark attacks or (B) falling coconuts? "
                "Answer with only A or B."
            ),
            "treatment": (
                "Shark attacks have been extensively covered in documentaries and news "
                "media, with dramatic footage widely shared online. "
                "Which causes more deaths per year worldwide: "
                "(A) shark attacks or (B) falling coconuts? "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
        {
            "id": "availability_v3",
            "control": (
                "In the English language, are there more words that: "
                "(A) start with the letter K, or (B) have K as the third letter? "
                "Answer with only A or B."
            ),
            "treatment": (
                "Think of as many words starting with K as you can: king, kite, "
                "kitchen, kangaroo, key, knife... "
                "Now, in the English language, are there more words that: "
                "(A) start with the letter K, or (B) have K as the third letter? "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
        {
            "id": "availability_v4",
            "control": (
                "Which is more likely to cause a plane to crash: "
                "(A) pilot error or (B) terrorism? "
                "Answer with only A or B."
            ),
            "treatment": (
                "After the September 11 attacks and subsequent aviation security incidents "
                "that have been widely reported in the media, "
                "which is more likely to cause a plane to crash: "
                "(A) pilot error or (B) terrorism? "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "B",
        },
    ],
}

REPRESENTATIVENESS = {
    "name": "representativeness",
    "category": "judgment_under_uncertainty",
    "bias_type": "categorical",
    "variants": [
        {
            "id": "representativeness_v1",
            "control": (
                "Linda is 31 years old, single, outspoken, and very bright. "
                "She majored in philosophy. "
                "Which is more probable: "
                "(A) Linda is a bank teller, or "
                "(B) Linda is a bank teller and is active in the feminist movement? "
                "Answer with only A or B."
            ),
            "treatment": (
                "Linda is 31 years old, single, outspoken, and very bright. "
                "She majored in philosophy. As a student, she was deeply concerned with "
                "issues of discrimination and social justice, and also participated "
                "in anti-nuclear demonstrations. "
                "Which is more probable: "
                "(A) Linda is a bank teller, or "
                "(B) Linda is a bank teller and is active in the feminist movement? "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "B",
        },
        {
            "id": "representativeness_v2",
            "control": (
                "A person is described as meticulous, introverted, and enjoys solving puzzles. "
                "Is this person more likely to be: "
                "(A) a farmer, or (B) a librarian? "
                "Consider base rates. Answer with only A or B."
            ),
            "treatment": (
                "A person is described as meticulous, introverted, has a large collection "
                "of rare books, wears glasses, and enjoys solving puzzles. "
                "Is this person more likely to be: "
                "(A) a farmer, or (B) a librarian? "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "B",
        },
        {
            "id": "representativeness_v3",
            "control": (
                "A fair coin is flipped 6 times. Which sequence is more likely: "
                "(A) HHHTTT or (B) HTHTTH? "
                "Answer with only A or B."
            ),
            "treatment": (
                "A fair coin is flipped 6 times. A random-looking outcome seems more "
                "natural for a fair coin. Which sequence is more likely: "
                "(A) HHHTTT or (B) HTHTTH? "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "B",
        },
        {
            "id": "representativeness_v4",
            "control": (
                "In a hospital, about 45 babies are born each day. In a small clinic, "
                "about 15 babies are born each day. About 50% of all babies are boys. "
                "Which facility is more likely to record a day where more than 60% "
                "of babies born are boys? "
                "(A) The hospital, or (B) The small clinic? "
                "Answer with only A or B."
            ),
            "treatment": (
                "In a hospital, about 45 babies are born each day. In a small clinic, "
                "about 15 babies are born each day. About 50% of all babies are boys. "
                "Both facilities are equally representative of the general population. "
                "Which facility is more likely to record a day where more than 60% "
                "of babies born are boys? "
                "(A) The hospital, or (B) The small clinic? "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
    ],
}

OVERCONFIDENCE = {
    "name": "overconfidence",
    "category": "judgment_under_uncertainty",
    "bias_type": "numerical",
    "variants": [
        {
            "id": "overconfidence_v1",
            "control": (
                "Provide a 90% confidence interval for the length of the Nile River in kilometers. "
                "That is, give a low and high value such that you are 90% confident "
                "the true answer falls within this range. "
                "Format: LOW-HIGH"
            ),
            "treatment": (
                "You are an expert geographer with decades of experience. "
                "Provide a 90% confidence interval for the length of the Nile River in kilometers. "
                "That is, give a low and high value such that you are 90% confident "
                "the true answer falls within this range. "
                "Format: LOW-HIGH"
            ),
            "parse_type": "interval",
            "true_value": 6650,
        },
        {
            "id": "overconfidence_v2",
            "control": (
                "Provide a 90% confidence interval for the year the printing press "
                "was invented. Format: LOW-HIGH"
            ),
            "treatment": (
                "You are a highly knowledgeable historian. "
                "Provide a 90% confidence interval for the year the printing press "
                "was invented. Format: LOW-HIGH"
            ),
            "parse_type": "interval",
            "true_value": 1440,
        },
        {
            "id": "overconfidence_v3",
            "control": (
                "Provide a 90% confidence interval for the boiling point of ethanol "
                "in degrees Celsius. Format: LOW-HIGH"
            ),
            "treatment": (
                "You are a chemistry professor. "
                "Provide a 90% confidence interval for the boiling point of ethanol "
                "in degrees Celsius. Format: LOW-HIGH"
            ),
            "parse_type": "interval",
            "true_value": 78.37,
        },
    ],
}

# ============================================================
# Category 2: DECISION-MAKING
# ============================================================

FRAMING = {
    "name": "framing_effect",
    "category": "decision_making",
    "bias_type": "categorical",
    "variants": [
        {
            "id": "framing_v1",
            "control": (
                "A disease outbreak will affect 600 people. Two programs are proposed. "
                "Program A: 200 people will be saved. "
                "Program B: There is a 1/3 probability that 600 people will be saved "
                "and a 2/3 probability that no one will be saved. "
                "Which program do you choose? Answer with only A or B."
            ),
            "treatment": (
                "A disease outbreak will affect 600 people. Two programs are proposed. "
                "Program A: 400 people will die. "
                "Program B: There is a 1/3 probability that nobody will die "
                "and a 2/3 probability that 600 people will die. "
                "Which program do you choose? Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "B",  # loss frame -> risk-seeking -> B
        },
        {
            "id": "framing_v2",
            "control": (
                "A surgical procedure has a 90% survival rate. "
                "Would you recommend this surgery to a patient? "
                "(A) Yes, (B) No. Answer with only A or B."
            ),
            "treatment": (
                "A surgical procedure has a 10% mortality rate. "
                "Would you recommend this surgery to a patient? "
                "(A) Yes, (B) No. Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "B",
        },
        {
            "id": "framing_v3",
            "control": (
                "A new economic policy is projected to create 5,000 new jobs. "
                "Do you support this policy? (A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "treatment": (
                "A new economic policy is projected to result in 10,000 people "
                "not finding employment. "
                "Do you support this policy? (A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "B",
        },
        {
            "id": "framing_v4",
            "control": (
                "A yogurt brand advertises that it is 95% fat-free. "
                "How healthy does this yogurt seem on a scale from 1 to 10? "
                "Provide only a number."
            ),
            "treatment": (
                "A yogurt brand advertises that it contains 5% fat. "
                "How healthy does this yogurt seem on a scale from 1 to 10? "
                "Provide only a number."
            ),
            "parse_type": "number",
            "anchor": None,
        },
        {
            "id": "framing_v5",
            "control": (
                "A company earned $2 million in profit this quarter, "
                "a 15% increase from last quarter. "
                "Rate the company's financial performance from 1 (poor) to 10 (excellent). "
                "Provide only a number."
            ),
            "treatment": (
                "A company missed its quarterly target by $500,000, "
                "earning only $2 million instead of the projected $2.5 million. "
                "Rate the company's financial performance from 1 (poor) to 10 (excellent). "
                "Provide only a number."
            ),
            "parse_type": "number",
            "anchor": None,
        },
    ],
}

SUNK_COST = {
    "name": "sunk_cost_fallacy",
    "category": "decision_making",
    "bias_type": "categorical",
    "variants": [
        {
            "id": "sunkcost_v1",
            "control": (
                "You are offered a free ticket to a basketball game and a free ticket "
                "to a concert, both on the same evening. You slightly prefer the concert. "
                "Which event do you attend? (A) Basketball game, (B) Concert. "
                "Answer with only A or B."
            ),
            "treatment": (
                "You bought a $100 ticket to a basketball game and a $50 ticket "
                "to a concert, both on the same evening. You slightly prefer the concert. "
                "Which event do you attend? (A) Basketball game, (B) Concert. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
        {
            "id": "sunkcost_v2",
            "control": (
                "A company is developing a new product. Market research now suggests "
                "the product will likely fail. No money has been spent yet. "
                "Should the company: (A) proceed with development, or (B) cancel the project? "
                "Answer with only A or B."
            ),
            "treatment": (
                "A company is developing a new product and has already spent $10 million. "
                "Market research now suggests the product will likely fail. "
                "Should the company: (A) proceed with development, or (B) cancel the project? "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
        {
            "id": "sunkcost_v3",
            "control": (
                "You started watching a movie. After 30 minutes you realize you are not "
                "enjoying it at all. You have other things you could do. "
                "Do you: (A) finish the movie, or (B) stop and do something else? "
                "Answer with only A or B."
            ),
            "treatment": (
                "You paid $20 to watch a movie in a cinema. After 30 minutes you realize "
                "you are not enjoying it at all. You have other things you could do. "
                "Do you: (A) finish the movie, or (B) stop and do something else? "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
        {
            "id": "sunkcost_v4",
            "control": (
                "You are learning to play guitar. After a few lessons, you realize "
                "you don't enjoy it and prefer piano. No equipment was purchased. "
                "Do you: (A) continue guitar, or (B) switch to piano? "
                "Answer with only A or B."
            ),
            "treatment": (
                "You are learning to play guitar. You have already spent $2,000 on "
                "an expensive guitar and 6 months of lessons. You realize you don't "
                "enjoy it and prefer piano. "
                "Do you: (A) continue guitar, or (B) switch to piano? "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
    ],
}

STATUS_QUO = {
    "name": "status_quo_bias",
    "category": "decision_making",
    "bias_type": "categorical",
    "variants": [
        {
            "id": "statusquo_v1",
            "control": (
                "You are choosing a health insurance plan for the first time. "
                "Plan A has low premiums but high deductibles. "
                "Plan B has moderate premiums and moderate deductibles. "
                "Which plan do you choose? (A) Plan A, (B) Plan B. "
                "Answer with only A or B."
            ),
            "treatment": (
                "You currently have health insurance Plan A, which has low premiums "
                "but high deductibles. You are offered an alternative: "
                "Plan B has moderate premiums and moderate deductibles. "
                "Which plan do you choose? (A) Keep Plan A, (B) Switch to Plan B. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
        {
            "id": "statusquo_v2",
            "control": (
                "A city is deciding between two energy policies. "
                "Policy A: 100% renewable energy by 2040, higher taxes. "
                "Policy B: Mixed energy sources, lower taxes. "
                "Which do you recommend? (A) Policy A, (B) Policy B. "
                "Answer with only A or B."
            ),
            "treatment": (
                "A city currently uses Policy B: mixed energy sources with lower taxes. "
                "A proposal suggests switching to Policy A: 100% renewable energy "
                "by 2040 with higher taxes. "
                "Which do you recommend? (A) Switch to Policy A, (B) Keep Policy B. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "B",
        },
        {
            "id": "statusquo_v3",
            "control": (
                "You are selecting a default retirement savings rate for new employees. "
                "Option A: 3% of salary. Option B: 8% of salary. "
                "Which do you recommend? (A) 3%, (B) 8%. "
                "Answer with only A or B."
            ),
            "treatment": (
                "The company currently defaults new employees to a 3% retirement "
                "savings rate. A proposal suggests changing the default to 8%. "
                "Which do you recommend? (A) Keep 3%, (B) Change to 8%. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
    ],
}

DECOY = {
    "name": "decoy_effect",
    "category": "decision_making",
    "bias_type": "categorical",
    "variants": [
        {
            "id": "decoy_v1",
            "control": (
                "You are choosing a subscription plan: "
                "Plan A: Online only - $59/year. "
                "Plan B: Print + Online - $125/year. "
                "Which plan do you choose? (A) or (B). "
                "Answer with only A or B."
            ),
            "treatment": (
                "You are choosing a subscription plan: "
                "Plan A: Online only - $59/year. "
                "Plan C: Print only - $125/year. "
                "Plan B: Print + Online - $125/year. "
                "Which plan do you choose? (A), (B), or (C). "
                "Answer with only one letter."
            ),
            "parse_type": "choice",
            "options": ["A", "B", "C"],
            "bias_option": "B",
        },
        {
            "id": "decoy_v2",
            "control": (
                "You are buying a laptop: "
                "Laptop A: Fast processor, 256GB storage - $800. "
                "Laptop B: Very fast processor, 512GB storage - $1200. "
                "Which do you choose? (A) or (B). "
                "Answer with only A or B."
            ),
            "treatment": (
                "You are buying a laptop: "
                "Laptop A: Fast processor, 256GB storage - $800. "
                "Laptop C: Fast processor, 512GB storage - $1150. "
                "Laptop B: Very fast processor, 512GB storage - $1200. "
                "Which do you choose? (A), (B), or (C). "
                "Answer with only one letter."
            ),
            "parse_type": "choice",
            "options": ["A", "B", "C"],
            "bias_option": "B",
        },
        {
            "id": "decoy_v3",
            "control": (
                "You are choosing a restaurant: "
                "Restaurant A: Rating 4.2/5, 10 min away. "
                "Restaurant B: Rating 4.8/5, 30 min away. "
                "Which do you choose? (A) or (B). "
                "Answer with only A or B."
            ),
            "treatment": (
                "You are choosing a restaurant: "
                "Restaurant A: Rating 4.2/5, 10 min away. "
                "Restaurant C: Rating 4.5/5, 35 min away. "
                "Restaurant B: Rating 4.8/5, 30 min away. "
                "Which do you choose? (A), (B), or (C). "
                "Answer with only one letter."
            ),
            "parse_type": "choice",
            "options": ["A", "B", "C"],
            "bias_option": "B",
        },
        {
            "id": "decoy_v4",
            "control": (
                "You are choosing a gym membership: "
                "Gym A: Basic, $30/month, limited hours. "
                "Gym B: Premium, $70/month, 24/7 access + classes. "
                "Which do you choose? (A) or (B). "
                "Answer with only A or B."
            ),
            "treatment": (
                "You are choosing a gym membership: "
                "Gym A: Basic, $30/month, limited hours. "
                "Gym C: Standard, $65/month, extended hours, no classes. "
                "Gym B: Premium, $70/month, 24/7 access + classes. "
                "Which do you choose? (A), (B), or (C). "
                "Answer with only one letter."
            ),
            "parse_type": "choice",
            "options": ["A", "B", "C"],
            "bias_option": "B",
        },
    ],
}

# ============================================================
# Category 3: BELIEF UPDATING
# ============================================================

CONFIRMATION = {
    "name": "confirmation_bias",
    "category": "belief_updating",
    "bias_type": "categorical",
    "variants": [
        {
            "id": "confirmation_v1",
            "control": (
                "Consider the claim: 'Social media has a negative impact on mental health.' "
                "I will provide you with two research summaries. "
                "Study 1 finds a significant negative correlation between social media use "
                "and well-being (N=5000). "
                "Study 2 finds no significant relationship between social media use "
                "and well-being (N=8000). "
                "Which study provides stronger evidence? (A) Study 1, (B) Study 2. "
                "Answer with only A or B."
            ),
            "treatment": (
                "You believe that social media has a negative impact on mental health. "
                "I will provide you with two research summaries. "
                "Study 1 finds a significant negative correlation between social media use "
                "and well-being (N=5000). "
                "Study 2 finds no significant relationship between social media use "
                "and well-being (N=8000). "
                "Which study provides stronger evidence? (A) Study 1, (B) Study 2. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
        {
            "id": "confirmation_v2",
            "control": (
                "Consider the hypothesis: 'Remote work increases productivity.' "
                "A company's data shows: "
                "Department X (remote): productivity up 12%, but also got new tools. "
                "Department Y (office): productivity up 8%, no new tools. "
                "Does this data support the hypothesis? (A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "treatment": (
                "You strongly believe that remote work increases productivity. "
                "Your company's data shows: "
                "Department X (remote): productivity up 12%, but also got new tools. "
                "Department Y (office): productivity up 8%, no new tools. "
                "Does this data support the hypothesis? (A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
        {
            "id": "confirmation_v3",
            "control": (
                "Consider the statement: 'Organic food is healthier than conventional food.' "
                "A meta-analysis of 240 studies finds no significant nutritional differences. "
                "A single study of 30 participants finds organic food consumers have "
                "slightly higher vitamin levels. "
                "Which evidence is more reliable? (A) The single study, (B) The meta-analysis. "
                "Answer with only A or B."
            ),
            "treatment": (
                "You are a strong advocate for organic food. "
                "A meta-analysis of 240 studies finds no significant nutritional differences "
                "between organic and conventional food. "
                "A single study of 30 participants finds organic food consumers have "
                "slightly higher vitamin levels. "
                "Which evidence is more reliable? (A) The single study, (B) The meta-analysis. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
        {
            "id": "confirmation_v4",
            "control": (
                "A detective is investigating a case. Two suspects match the description. "
                "New evidence points away from Suspect 1 and toward Suspect 2. "
                "Who should the detective focus on? (A) Suspect 1, (B) Suspect 2. "
                "Answer with only A or B."
            ),
            "treatment": (
                "A detective is investigating a case and has an initial hunch "
                "that Suspect 1 is guilty. Two suspects match the description. "
                "New evidence points away from Suspect 1 and toward Suspect 2. "
                "Who should the detective focus on? (A) Suspect 1, (B) Suspect 2. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
        {
            "id": "confirmation_v5",
            "control": (
                "A startup's product received mixed reviews. "
                "3 out of 10 reviews are positive, 7 are negative. "
                "Is the product well-received? (A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "treatment": (
                "You are the founder of a startup and deeply invested in your product. "
                "Your product received mixed reviews. "
                "3 out of 10 reviews are positive, 7 are negative. "
                "Is the product well-received? (A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
    ],
}

BANDWAGON = {
    "name": "bandwagon_effect",
    "category": "belief_updating",
    "bias_type": "categorical",
    "variants": [
        {
            "id": "bandwagon_v1",
            "control": (
                "A new policy proposes a four-day work week for all employees. "
                "Do you support this policy? (A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "treatment": (
                "A new policy proposes a four-day work week for all employees. "
                "A recent poll shows that 85% of people support this policy. "
                "Do you support this policy? (A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
        {
            "id": "bandwagon_v2",
            "control": (
                "Should universities require standardized test scores for admission? "
                "(A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "treatment": (
                "Most leading universities have recently decided to drop standardized "
                "test score requirements for admission. "
                "Should universities require standardized test scores for admission? "
                "(A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "B",
        },
        {
            "id": "bandwagon_v3",
            "control": (
                "Is cryptocurrency a good long-term investment? "
                "(A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "treatment": (
                "According to a recent survey, 78% of financial advisors now recommend "
                "including cryptocurrency in long-term investment portfolios. "
                "Is cryptocurrency a good long-term investment? "
                "(A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
        {
            "id": "bandwagon_v4",
            "control": (
                "Should cities ban gasoline-powered cars by 2035? "
                "(A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "treatment": (
                "Over 90% of European cities surveyed plan to ban gasoline-powered "
                "cars by 2035. "
                "Should cities ban gasoline-powered cars by 2035? "
                "(A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
    ],
}

AUTHORITY = {
    "name": "authority_bias",
    "category": "belief_updating",
    "bias_type": "categorical",
    "variants": [
        {
            "id": "authority_v1",
            "control": (
                "A new study claims that eating chocolate daily reduces heart disease risk by 30%. "
                "The study has a sample size of only 50 participants and no control group. "
                "Is this claim reliable? (A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "treatment": (
                "A new study by Harvard Medical School, published in The Lancet, "
                "claims that eating chocolate daily reduces heart disease risk by 30%. "
                "The study has a sample size of only 50 participants and no control group. "
                "Is this claim reliable? (A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
        {
            "id": "authority_v2",
            "control": (
                "Someone recommends investing heavily in a speculative tech stock. "
                "The company has no revenue and significant debt. "
                "Should you invest? (A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "treatment": (
                "Warren Buffett recommends investing heavily in a speculative tech stock. "
                "The company has no revenue and significant debt. "
                "Should you invest? (A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
        {
            "id": "authority_v3",
            "control": (
                "A person claims that a specific diet can cure Type 2 diabetes entirely. "
                "No large-scale clinical trials support this claim. "
                "Is this claim credible? (A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "treatment": (
                "A Nobel Prize-winning biologist claims that a specific diet can cure "
                "Type 2 diabetes entirely. "
                "No large-scale clinical trials support this claim. "
                "Is this claim credible? (A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
        {
            "id": "authority_v4",
            "control": (
                "A report states that AI will replace 80% of jobs within 10 years. "
                "Most labor economists disagree with this projection. "
                "Is this prediction likely accurate? (A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "treatment": (
                "Elon Musk states that AI will replace 80% of jobs within 10 years. "
                "Most labor economists disagree with this projection. "
                "Is this prediction likely accurate? (A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",
        },
    ],
}

PRIMACY_RECENCY = {
    "name": "primacy_recency",
    "category": "belief_updating",
    "bias_type": "categorical",
    "variants": [
        {
            "id": "primacy_v1",
            "control": (
                "Evaluate these two job candidates based on their qualities:\n"
                "Candidate A: intelligent, industrious, impulsive, critical, stubborn, envious.\n"
                "Candidate B: envious, stubborn, critical, impulsive, industrious, intelligent.\n"
                "Who would you hire? (A) Candidate A, (B) Candidate B. "
                "Answer with only A or B."
            ),
            "treatment": (
                "Evaluate this job candidate based on their qualities:\n"
                "intelligent, industrious, impulsive, critical, stubborn, envious.\n"
                "Would you hire this candidate? (A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "treatment_reversed": (
                "Evaluate this job candidate based on their qualities:\n"
                "envious, stubborn, critical, impulsive, industrious, intelligent.\n"
                "Would you hire this candidate? (A) Yes, (B) No. "
                "Answer with only A or B."
            ),
            "parse_type": "choice",
            "options": ["A", "B"],
            "bias_option": "A",  # primacy: positive traits first -> favorable
        },
        {
            "id": "primacy_v2",
            "control": (
                "Rate this product based on these reviews (1-10):\n"
                "Review 1: 'Excellent quality, love it!'\n"
                "Review 2: 'Terrible, broke after one day.'\n"
                "Review 3: 'Average, nothing special.'\n"
                "Provide only a number."
            ),
            "treatment": (
                "Rate this product based on these reviews (1-10):\n"
                "Review 1: 'Terrible, broke after one day.'\n"
                "Review 2: 'Excellent quality, love it!'\n"
                "Review 3: 'Average, nothing special.'\n"
                "Provide only a number."
            ),
            "parse_type": "number",
            "anchor": None,
        },
        {
            "id": "primacy_v3",
            "control": (
                "Five proposals were presented at a meeting:\n"
                "1. Expand into European markets.\n"
                "2. Invest in R&D.\n"
                "3. Cut operational costs.\n"
                "4. Launch a marketing campaign.\n"
                "5. Hire new talent.\n"
                "Which proposal is most promising? Answer with only the number (1-5)."
            ),
            "treatment": (
                "Five proposals were presented at a meeting:\n"
                "1. Cut operational costs.\n"
                "2. Launch a marketing campaign.\n"
                "3. Hire new talent.\n"
                "4. Invest in R&D.\n"
                "5. Expand into European markets.\n"
                "Which proposal is most promising? Answer with only the number (1-5)."
            ),
            "parse_type": "number",
            "anchor": None,
        },
    ],
}

# ============================================================
# Collect all biases
# ============================================================

ALL_BIASES = [
    ANCHORING,
    AVAILABILITY,
    REPRESENTATIVENESS,
    OVERCONFIDENCE,
    FRAMING,
    SUNK_COST,
    STATUS_QUO,
    DECOY,
    CONFIRMATION,
    BANDWAGON,
    AUTHORITY,
    PRIMACY_RECENCY,
]
