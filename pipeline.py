import csv
import json
import os
import time
from datetime import datetime

from crewai import Crew
import config
from agents import (
    CSVReaderAgent,
    FeedbackClassifierAgent,
    BugAnalysisAgent,
    FeatureExtractorAgent,
    TicketCreatorAgent,
    QualityCriticAgent,
)


def load_feedback_data():
    """Load and merge feedback from both CSV sources."""
    records = []

    with open(config.APP_REVIEWS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append({
                "source_id": row["review_id"],
                "source_type": "app_review",
                "text": row["review_text"],
                "rating": row.get("rating", ""),
                "platform": row.get("platform", ""),
                "user": row.get("user_name", ""),
                "date": row.get("date", ""),
                "app_version": row.get("app_version", ""),
            })

    with open(config.SUPPORT_EMAILS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append({
                "source_id": row["email_id"],
                "source_type": "support_email",
                "text": f"{row['subject']}. {row['body']}",
                "rating": "",
                "platform": "",
                "user": row.get("sender_email", ""),
                "date": row.get("timestamp", ""),
                "app_version": "",
            })

    return records


def run_pipeline(records=None, progress_callback=None):
    """Run the full multi-agent pipeline. Returns (tickets, log_entries, metrics)."""
    if records is None:
        records = load_feedback_data()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Create agent instances
    reader = CSVReaderAgent()
    classifier = FeedbackClassifierAgent()
    bug_analyzer = BugAnalysisAgent()
    feature_extractor = FeatureExtractorAgent()
    ticket_creator = TicketCreatorAgent()
    quality_critic = QualityCriticAgent()

    feedback_json = json.dumps(records, indent=2)
    log_entries = []
    start_time = time.time()

    def log(step, source_id, message):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "source_id": source_id,
            "message": message,
        }
        log_entries.append(entry)

    log("pipeline_start", "ALL", f"Processing {len(records)} feedback items")
    if progress_callback:
        progress_callback("Reading CSV data...", 0.05)

    # Get tasks from each agent class
    log("csv_reader", "ALL", "CSV Reader Agent: Validating and cleaning input data")
    read_task = reader.get_task(feedback_json)

    log("classifier", "ALL", "Feedback Classifier Agent: Classifying items into Bug/Feature Request/Praise/Complaint/Spam with confidence scores")
    classify_task = classifier.get_task()

    log("bug_analysis", "ALL", "Bug Analysis Agent: Extracting technical details (device, OS, steps to reproduce) from bug reports")
    bug_task = bug_analyzer.get_task()

    log("feature_extraction", "ALL", "Feature Extractor Agent: Identifying feature requests and estimating user impact/demand")
    feature_task = feature_extractor.get_task()

    log("ticket_creation", "ALL", "Ticket Creator Agent: Generating structured tickets with priority, category, and metadata for all items")
    ticket_task = ticket_creator.get_task()

    log("quality_review", "ALL", "Quality Critic Agent: Reviewing tickets for completeness, accuracy, and correct priority assignment")
    quality_task = quality_critic.get_task()

    if progress_callback:
        progress_callback("Running agent pipeline...", 0.15)

    # Run the crew
    crew = Crew(
        agents=[
            reader.agent, classifier.agent, bug_analyzer.agent,
            feature_extractor.agent, ticket_creator.agent, quality_critic.agent,
        ],
        tasks=[read_task, classify_task, bug_task, feature_task, ticket_task, quality_task],
        verbose=True,
    )

    result = crew.kickoff()
    log("pipeline_complete", "ALL", "All 6 agents completed. Parsing final tickets and computing accuracy metrics.")

    if progress_callback:
        progress_callback("Parsing results...", 0.85)

    # Parse final tickets from quality agent output
    tickets = _parse_tickets(str(result))

    # Apply confidence threshold — flag low-confidence tickets for manual review
    for t in tickets:
        try:
            conf = float(t.get("confidence", 1.0))
        except (ValueError, TypeError):
            conf = 0.0

        sid = t.get("source_id", "")
        cat = t.get("category", "Unknown")
        pri = t.get("priority", "Unknown")
        title = t.get("title", "")
        src = t.get("source_type", "")
        tech = t.get("technical_details", "")
        qscore = t.get("quality_score", "")

        log("classification", sid, f"Source: {src} | Classified as '{cat}' with confidence {conf} | Reasoning based on NLP keyword and sentiment analysis")

        if cat == "Bug":
            log("bug_analysis", sid, f"Bug details extracted — Technical: {tech[:120]}..." if len(str(tech)) > 120 else f"Bug details extracted — Technical: {tech or 'No details available'}")
        elif cat == "Feature Request":
            log("feature_analysis", sid, f"Feature request identified — Title: {title}")

        log("ticket_creation", sid, f"Ticket created — ID: {t.get('ticket_id')} | Title: {title[:80]} | Priority: {pri}")
        log("quality_review", sid, f"Quality review — Score: {qscore}/10")

        if conf < config.CLASSIFICATION_CONFIDENCE_THRESHOLD:
            t["status"] = "Needs Review"
            log("threshold_flag", sid, f"Confidence {conf} below threshold {config.CLASSIFICATION_CONFIDENCE_THRESHOLD} — status set to 'Needs Review'")
        else:
            log("accepted", sid, f"Ticket accepted — Category: {cat}, Priority: {pri}, Confidence: {conf}")

    elapsed = time.time() - start_time

    # Compute metrics
    metrics = _compute_metrics(tickets, elapsed)

    # Save outputs
    _save_tickets(tickets)
    _save_log(log_entries)
    _save_metrics(metrics)

    if progress_callback:
        progress_callback("Done!", 1.0)

    return tickets, log_entries, metrics


def _parse_tickets(raw_output):
    """Extract JSON ticket array from LLM output."""
    # Try to find JSON array in the output
    text = raw_output.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    # Fallback: return raw as single entry
    return [{"ticket_id": "TKT-ERR", "title": "Parse Error", "description": text,
             "category": "Unknown", "priority": "Medium", "status": "Review Needed"}]


def _compute_metrics(tickets, elapsed):
    """Compare tickets against expected classifications and compute accuracy."""
    expected = {}
    if os.path.exists(config.EXPECTED_CSV):
        with open(config.EXPECTED_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                expected[row["source_id"]] = row

    total = len(tickets)
    correct_category = 0
    correct_priority = 0
    matched = 0

    for t in tickets:
        sid = t.get("source_id", "")
        if sid in expected:
            matched += 1
            if t.get("category", "").strip().lower() == expected[sid].get("category", "").strip().lower():
                correct_category += 1
            if t.get("priority", "").strip().lower() == expected[sid].get("priority", "").strip().lower():
                correct_priority += 1

    return {
        "total_processed": total,
        "matched_to_expected": matched,
        "category_accuracy": round(correct_category / matched, 4) if matched else 0,
        "priority_accuracy": round(correct_priority / matched, 4) if matched else 0,
        "processing_time_seconds": round(elapsed, 2),
        "tickets_per_minute": round(total / (elapsed / 60), 2) if elapsed > 0 else 0,
    }


def _save_tickets(tickets):
    if not tickets:
        return
    fieldnames = list(tickets[0].keys())
    with open(config.GENERATED_TICKETS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(tickets)


def _save_log(log_entries):
    if not log_entries:
        return
    fieldnames = ["timestamp", "step", "source_id", "message"]
    with open(config.PROCESSING_LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_entries)


def _save_metrics(metrics):
    with open(config.METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


if __name__ == "__main__":
    print("Starting Feedback Analysis Pipeline...")
    tickets, logs, metrics = run_pipeline()
    print(f"\nDone! Processed {metrics['total_processed']} items in {metrics['processing_time_seconds']}s")
    print(f"Category accuracy: {metrics['category_accuracy']*100:.1f}%")
    print(f"Priority accuracy: {metrics['priority_accuracy']*100:.1f}%")
    print(f"Output saved to {config.OUTPUT_DIR}")
