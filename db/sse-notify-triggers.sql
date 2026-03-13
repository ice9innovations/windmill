-- SSE LISTEN/NOTIFY triggers for ice9-api /stream endpoint
--
-- These triggers fire pg_notify('ice9_result', image_id) whenever a result
-- row is written, allowing ice9-api's SSE stream endpoint to wake up
-- immediately rather than polling the DB.
--
-- Idempotent — safe to re-run.

CREATE OR REPLACE FUNCTION notify_result_change()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('ice9_result', NEW.image_id::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Primary service results (INSERT only — each result is a new row)
DROP TRIGGER IF EXISTS results_notify ON results;
CREATE TRIGGER results_notify
    AFTER INSERT ON results
    FOR EACH ROW EXECUTE FUNCTION notify_result_change();

-- content_analysis (ON CONFLICT DO UPDATE)
DROP TRIGGER IF EXISTS content_analysis_notify ON content_analysis;
CREATE TRIGGER content_analysis_notify
    AFTER INSERT OR UPDATE ON content_analysis
    FOR EACH ROW EXECUTE FUNCTION notify_result_change();

-- consensus (DELETE + INSERT — INSERT trigger is sufficient)
DROP TRIGGER IF EXISTS consensus_notify ON consensus;
CREATE TRIGGER consensus_notify
    AFTER INSERT ON consensus
    FOR EACH ROW EXECUTE FUNCTION notify_result_change();

-- noun_consensus (ON CONFLICT DO UPDATE)
DROP TRIGGER IF EXISTS noun_consensus_notify ON noun_consensus;
CREATE TRIGGER noun_consensus_notify
    AFTER INSERT OR UPDATE ON noun_consensus
    FOR EACH ROW EXECUTE FUNCTION notify_result_change();

-- verb_consensus (ON CONFLICT DO UPDATE)
DROP TRIGGER IF EXISTS verb_consensus_notify ON verb_consensus;
CREATE TRIGGER verb_consensus_notify
    AFTER INSERT OR UPDATE ON verb_consensus
    FOR EACH ROW EXECUTE FUNCTION notify_result_change();

-- caption_summary (ON CONFLICT DO UPDATE)
DROP TRIGGER IF EXISTS caption_summary_notify ON caption_summary;
CREATE TRIGGER caption_summary_notify
    AFTER INSERT OR UPDATE ON caption_summary
    FOR EACH ROW EXECUTE FUNCTION notify_result_change();

-- postprocessing (INSERT only — colors_post, caption_score_*, face, pose)
DROP TRIGGER IF EXISTS postprocessing_notify ON postprocessing;
CREATE TRIGGER postprocessing_notify
    AFTER INSERT ON postprocessing
    FOR EACH ROW EXECUTE FUNCTION notify_result_change();
