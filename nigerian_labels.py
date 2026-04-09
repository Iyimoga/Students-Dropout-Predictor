"""
Nigerian-context mappings for the dropout model.
Every user-friendly label maps to the numeric code the model was trained on.
"""

# ---------- BINARY ----------
YES_NO = {"No": 0, "Yes": 1}
GENDER = {"Female": 0, "Male": 1}

# ---------- MARITAL STATUS (codes 1-6) ----------
MARITAL_STATUS = {
    "Single": 1,
    "Married": 2,
    "Widowed": 3,
    "Divorced": 4,
    "Living together (common-law)": 5,
    "Separated": 6,
}

# ---------- APPLICATION MODE (how you got admission) ----------
# Mapped to Nigerian admission pathways. Codes are valid 1-18 from training data.
APPLICATION_MODE = {
    "UTME / JAMB — 1st choice": 1,
    "UTME / JAMB — 2nd choice": 2,
    "Post-UTME (supplementary)": 3,
    "Direct Entry (A-Level / IJMB)": 4,
    "Direct Entry (NCE / OND / HND)": 5,
    "Transfer from another university": 6,
    "Change of course (inter-faculty)": 7,
    "Mature student (above 23 years)": 8,
    "Pre-degree / Foundation programme": 9,
    "Diploma holder conversion": 10,
    "International / Foreign student": 11,
    "Part-time / Sandwich programme": 12,
    "Distance learning": 13,
    "Scholarship placement": 14,
    "Catchment area / ELDS quota": 15,
    "Staff dependant quota": 16,
    "Re-admission after withdrawal": 17,
    "Other": 18,
}

# ---------- PREVIOUS QUALIFICATION (codes 1-17) ----------
PREVIOUS_QUALIFICATION = {
    "SSCE / WAEC / NECO (secondary school)": 1,
    "GCE / IGCSE": 2,
    "NABTEB (technical secondary)": 3,
    "NCE (Nigeria Certificate in Education)": 4,
    "OND (Ordinary National Diploma)": 5,
    "HND (Higher National Diploma)": 6,
    "Bachelor's degree": 7,
    "Postgraduate Diploma": 8,
    "Master's degree": 9,
    "PhD / Doctorate": 10,
    "IJMB / JUPEB (A-Level equivalent)": 11,
    "Cambridge / Edexcel A-Levels": 12,
    "Pre-degree / Foundation certificate": 13,
    "Professional certification (ICAN, CIBN, etc.)": 14,
    "Diploma in Theology / Law / Nursing": 15,
    "Incomplete secondary education": 16,
    "Other": 17,
}

# ---------- COURSE (codes 1-17) ----------
# Mapped to common Nigerian undergraduate programmes
COURSE = {
    "Agriculture / Agronomy": 1,
    "Mass Communication / Multimedia": 2,
    "Social Work (Part-time)": 3,
    "Crop Science / Soil Science": 4,
    "Graphic Design / Fine Arts": 5,
    "Veterinary Medicine": 6,
    "Computer Science / Engineering": 7,
    "Animal Science": 8,
    "Business Administration / Management": 9,
    "Social Work / Sociology": 10,
    "Tourism / Hospitality Management": 11,
    "Nursing Science": 12,
    "Dental Therapy / Oral Hygiene": 13,
    "Marketing / Advertising": 14,
    "Journalism / Public Relations": 15,
    "Education (B.Ed)": 16,
    "Business Administration (Part-time)": 17,
}

# ---------- PARENT EDUCATION (simplified buckets) ----------
# Each Nigerian label maps to a representative code from the training data
PARENT_QUALIFICATION = {
    "No formal education": 19,
    "Primary school (Basic 1–6)": 18,
    "Junior Secondary (JSS)": 15,
    "Senior Secondary (SSCE / WAEC)": 1,
    "NCE / OND (Technical / Teaching)": 4,
    "HND / Bachelor's degree": 2,
    "Master's degree": 3,
    "PhD / Doctorate": 5,
    "Unknown": 22,
}

# ---------- PARENT OCCUPATION (simplified) ----------
PARENT_OCCUPATION = {
    "Unemployed / Not working": 1,
    "Farmer / Fisherman": 7,
    "Petty trader / Market seller": 6,
    "Artisan (tailor, mechanic, carpenter, etc.)": 8,
    "Civil servant (junior / administrative)": 5,
    "Teacher / Lecturer": 4,
    "Professional (doctor, engineer, lawyer, accountant)": 3,
    "Business owner / Entrepreneur": 2,
    "Driver / Machine operator": 9,
    "Unskilled worker (cleaner, security, labourer)": 10,
    "Retired / Pensioner": 11,
    "Deceased": 12,
    "Other / Unknown": 32,
}

# ---------- GRADE SCALE HELPERS ----------
# Training data uses Portuguese 0–20 scale. Nigerian students understand
# percentages (0–100) and CGPA (0–5). We let them enter percentage.
def percentage_to_portuguese(pct: float) -> float:
    """Convert Nigerian percentage (0–100) to Portuguese 0–20 grade."""
    return round(pct / 5.0, 2)

def nigerian_grade_label(pct: float) -> str:
    if pct >= 70:   return "A (Excellent)"
    if pct >= 60:   return "B (Very Good)"
    if pct >= 50:   return "C (Good)"
    if pct >= 45:   return "D (Pass)"
    if pct >= 40:   return "E (Weak Pass)"
    return "F (Fail)"
