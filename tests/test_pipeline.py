"""Unit tests for the Support Ticket Triage pipeline."""

import unittest
import pandas as pd
from src.preprocessing import clean_text_field, strip_email_boilerplate, detect_text_degraded
from src.classifier import TicketClassifier
from src.triage import TriageEngine
from src.utils import VALID_CATEGORIES, VALID_PRIORITIES, validate_output_dataframe


class TestTicketPipeline(unittest.TestCase):

    def test_clean_text_field(self):
        self.assertEqual(clean_text_field(None), "")
        self.assertEqual(clean_text_field(float("nan")), "")
        self.assertEqual(clean_text_field("  Hello \r\n world!  "), "Hello \n world!")

    def test_strip_email_boilerplate(self):
        raw = "Hello support,\n\nI need an invoice copy.\n\nKind regards,\nJohn"
        cleaned = strip_email_boilerplate(raw)
        self.assertNotIn("Hello support", cleaned)
        self.assertNotIn("Kind regards", cleaned)
        self.assertIn("I need an invoice copy.", cleaned)

    def test_detect_text_degraded(self):
        # Gibberish
        is_deg, _ = detect_text_degraded("???", "asdf asdf test test 123")
        self.assertTrue(is_deg)

        # Extremely short vague text
        is_deg, _ = detect_text_degraded("help", "help")
        self.assertTrue(is_deg)

        # Missing body and no subject
        is_deg, _ = detect_text_degraded("(no subject)", "")
        self.assertTrue(is_deg)

        # Valid ticket
        is_deg, _ = detect_text_degraded("Wrong item received", "I received EU plug version instead of UK.")
        self.assertFalse(is_deg)

    def test_classifier_categories(self):
        classifier = TicketClassifier()

        cases = [
            ({"clean_subject": "Invoice copy", "full_text": "Invoice copy. Send me invoice for LM-1234."}, "Billing"),
            ({"clean_subject": "Rechnung benötigt", "full_text": "Rechnung benötigt. Bitte Kopie der Rechnung senden."}, "Billing"),
            ({"clean_subject": "मेरे कार्ड से पैसे कट रहे हैं", "full_text": "मेरे कार्ड से पैसे कट रहे हैं।"}, "Billing"),
            ({"clean_subject": "Tracking not updating", "full_text": "Tracking not updating for order LM-555."}, "Shipping"),
            ({"clean_subject": "Wrong item received", "full_text": "Order LM-520106 arrived with EU plug."}, "Shipping"),
            ({"clean_subject": "गलत सामान मिला", "full_text": "गलत सामान मिला। मैंने दूसरा सामान मंगाया था।"}, "Shipping"),
            ({"clean_subject": "Password reset expired", "full_text": "The reset link expired before I could use it."}, "Account"),
            ({"clean_subject": "No puedo entrar", "full_text": "No puedo entrar a mi cuenta."}, "Account"),
            ({"clean_subject": "Lost my two-factor phone", "full_text": "My phone was stolen with authenticator app."}, "Account"),
            ({"clean_subject": "App crashes on start", "full_text": "App crashes immediately on launch."}, "Technical"),
            ({"clean_subject": "Devices not syncing", "full_text": "Changes on phone do not sync to tablet."}, "Technical"),
            ({"clean_subject": "アプリが遅い", "full_text": "アプリが遅い。動作が重い。"}, "Technical"),
            ({"clean_subject": "An idea for the app", "full_text": "Please add a dark mode to the application."}, "Other"),
            ({"clean_subject": "Great product", "full_text": "Just saying thanks, love the service."}, "Other"),
            ({"clean_subject": "Business enquiry", "full_text": "We want to become an authorized reseller."}, "Other"),
        ]

        for item, expected_cat in cases:
            res = classifier.predict(item)
            self.assertEqual(res["category"], expected_cat, f"Expected {expected_cat} for {item['clean_subject']}, got {res['category']}")

    def test_classifier_priorities(self):
        classifier = TicketClassifier()

        # Urgent cases
        urgent_ticket = {"clean_subject": "Unauthorised payments", "full_text": "Someone is using my saved card on your site."}
        res = classifier.predict(urgent_ticket)
        self.assertEqual(res["priority"], "Urgent")

        # High cases
        high_ticket = {"clean_subject": "Double charge on my card", "full_text": "I was charged twice for order LM-1234."}
        res = classifier.predict(high_ticket)
        self.assertEqual(res["priority"], "High")

        crash_ticket = {"clean_subject": "App crashes on start", "full_text": "The app crashes on launch every single time."}
        res = classifier.predict(crash_ticket)
        self.assertEqual(res["priority"], "High")

        # Low cases
        low_ticket = {"clean_subject": "Just saying thanks", "full_text": "Support agent was very helpful, thank you!"}
        res = classifier.predict(low_ticket)
        self.assertEqual(res["priority"], "Low")

        feature_ticket = {"clean_subject": "An idea for the app", "full_text": "Dark mode would be great."}
        res = classifier.predict(feature_ticket)
        self.assertEqual(res["priority"], "Low")

    def test_triage_decision_rules(self):
        triage = TriageEngine(confidence_threshold=0.65, preserve_tentative=False)

        # Confident ticket
        confident_pred = {
            "id": "T-0001",
            "category": "Billing",
            "category_confidence": 0.92,
            "category_margin": 0.45,
            "priority": "High",
            "priority_confidence": 0.90,
            "overall_confidence": 0.91,
            "is_degraded": False,
        }
        cat, pri, needs_review, _ = triage.evaluate_ticket(confident_pred)
        self.assertEqual(needs_review, "no")
        self.assertEqual(cat, "Billing")
        self.assertEqual(pri, "High")

        # Degraded ticket
        degraded_pred = {
            "id": "T-0007",
            "category": "Other",
            "category_confidence": 0.10,
            "category_margin": 0.0,
            "priority": "Low",
            "priority_confidence": 0.10,
            "overall_confidence": 0.10,
            "is_degraded": True,
            "degrade_reason": "Vague context ('help')",
        }
        cat, pri, needs_review, _ = triage.evaluate_ticket(degraded_pred)
        self.assertEqual(needs_review, "yes")
        self.assertEqual(cat, "")  # Empty per assignment spec
        self.assertEqual(pri, "")

        # Low confidence ticket
        low_conf_pred = {
            "id": "T-0100",
            "category": "Technical",
            "category_confidence": 0.40,
            "category_margin": 0.05,
            "priority": "Medium",
            "priority_confidence": 0.60,
            "overall_confidence": 0.35,
            "is_degraded": False,
        }
        cat, pri, needs_review, _ = triage.evaluate_ticket(low_conf_pred)
        self.assertEqual(needs_review, "yes")
        self.assertEqual(cat, "")
        self.assertEqual(pri, "")

    def test_output_dataframe_validation(self):
        valid_df = pd.DataFrame([
            {"id": "T-0001", "category": "Billing", "priority": "High", "needs_review": "no"},
            {"id": "T-0002", "category": "", "priority": "", "needs_review": "yes"},
        ])
        validate_output_dataframe(valid_df)

        invalid_df = pd.DataFrame([
            {"id": "T-0001", "category": "InvalidCat", "priority": "High", "needs_review": "no"}
        ])
        with self.assertRaises(ValueError):
            validate_output_dataframe(invalid_df)


if __name__ == "__main__":
    unittest.main()
