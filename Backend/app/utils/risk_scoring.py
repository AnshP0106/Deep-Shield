def calculate_risk_level(
    prediction: str,
    confidence: float
) -> str:

    confidence = float(confidence)

    # -----------------------------------------------------
    # Weak prediction = uncertain
    # -----------------------------------------------------

    if 45 <= confidence < 60:
        return "UNCERTAIN"

    # -----------------------------------------------------
    # FAKE prediction
    # -----------------------------------------------------

    if prediction == "FAKE":

        if confidence >= 80:
            return "VERY_HIGH"

        elif confidence >= 65:
            return "HIGH"

        else:
            return "MODERATE"

    # -----------------------------------------------------
    # REAL prediction
    # -----------------------------------------------------

    if prediction == "REAL":

        if confidence >= 80:
            return "LOW"

        elif confidence >= 65:
            return "MODERATE"

        else:
            return "UNCERTAIN"

    return "UNCERTAIN"


# =========================================================
# OVERALL IMAGE RISK
# =========================================================

def calculate_overall_risk(
    face_results: list
) -> dict:

    # -----------------------------------------------------
    # No faces
    # -----------------------------------------------------

    if not face_results:

        return {
            "risk_score": 0.0,
            "risk_level": "NO_FACE",
            "reason": "No faces were detected."
        }

    # -----------------------------------------------------
    # Extract face information
    # -----------------------------------------------------

    risk_values = []

    strong_fake_count = 0
    uncertain_count = 0
    real_count = 0

    for face in face_results:

        fake_probability = float(
            face["fake_probability"]
        )

        confidence = float(
            face["confidence"]
        )

        prediction = face["prediction"]

        # -------------------------------------------------
        # Fake probability contributes directly to risk
        # -------------------------------------------------

        risk_values.append(
            fake_probability
        )

        # -------------------------------------------------
        # Count categories
        # -------------------------------------------------

        if (
            prediction == "FAKE"
            and confidence >= 65
        ):

            strong_fake_count += 1

        elif (
            face.get("risk_level")
            == "UNCERTAIN"
        ):

            uncertain_count += 1

        elif prediction == "REAL":

            real_count += 1

    # -----------------------------------------------------
    # Base risk
    # -----------------------------------------------------

    average_risk = (
        sum(risk_values)
        / len(risk_values)
    )

    # -----------------------------------------------------
    # Strong fake evidence
    #
    # Give additional weight when multiple faces
    # strongly indicate manipulation.
    # -----------------------------------------------------

    fake_ratio = (
        strong_fake_count
        / len(face_results)
    )

    # Additional evidence from strong fake faces
    fake_evidence = (
        fake_ratio * 35.0
    )

    # -----------------------------------------------------
    # Uncertainty penalty
    #
    # Uncertain faces should slightly increase caution,
    # but should NOT automatically mean fake.
    # -----------------------------------------------------

    uncertainty_ratio = (
        uncertain_count
        / len(face_results)
    )

    uncertainty_penalty = (
        uncertainty_ratio * 10.0
    )

    # -----------------------------------------------------
    # Calculate final score
    # -----------------------------------------------------

    risk_score = (
        average_risk * 0.65
        + fake_evidence
        + uncertainty_penalty
    )

    # Keep between 0 and 100
    risk_score = max(
        0.0,
        min(
            100.0,
            risk_score
        )
    )

    # -----------------------------------------------------
    # Determine overall risk level
    # -----------------------------------------------------

    if risk_score >= 80:

        risk_level = "VERY_HIGH"

    elif risk_score >= 60:

        risk_level = "HIGH"

    elif risk_score >= 35:

        risk_level = "MODERATE"

    elif risk_score >= 15:

        risk_level = "LOW"

    else:

        risk_level = "VERY_LOW"

    # -----------------------------------------------------
    # Generate explanation
    # -----------------------------------------------------

    if strong_fake_count > 0:

        reason = (
            f"{strong_fake_count} face(s) showed "
            "strong evidence of manipulation."
        )

    elif uncertain_count > 0:

        reason = (
            f"{uncertain_count} face(s) produced "
            "uncertain model predictions."
        )

    else:

        reason = (
            "No strong evidence of facial "
            "manipulation was detected."
        )

    # -----------------------------------------------------
    # Return result
    # -----------------------------------------------------

    return {

        "risk_score": round(
            float(risk_score),
            2
        ),

        "risk_level":
            risk_level,

        "reason":
            reason,

        "faces_analyzed":
            len(face_results),

        "real_faces":
            real_count,

        "uncertain_faces":
            uncertain_count,

        "strong_fake_faces":
            strong_fake_count
    }