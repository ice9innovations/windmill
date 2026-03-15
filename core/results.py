"""
Result fetching from the windmill pipeline database.

No Flask dependency. Takes a psycopg2 RealDictCursor — the caller is responsible
for opening the connection and cursor.

NOTE on ice9-api usage: ice9-api adds per-account ownership filtering
(AND account_id = %s) to protect against IDOR. That is a business rule, not
pipeline logic. ice9-api keeps its own _fetch_results locally; this module is
not imported by ice9-api.
"""


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_results(cur, image_id):
    """Fetch all pipeline results for an image.

    Returns a dict with keys: service_results, merged_boxes, consensus,
    content_analysis, postprocessing, noun_consensus, sam3, verb_consensus,
    caption_summary, service_dispatch.

    See module docstring for ice9-api ownership-filtering caveat.
    """
    # Service results
    cur.execute(
        """SELECT service, data, status, processing_time, result_created
           FROM results WHERE image_id = %s AND status = 'success'
           ORDER BY result_created""",
        (image_id,),
    )
    service_results = {}
    for r in cur.fetchall():
        service_results[r['service']] = {
            "data":            r['data'],
            "processing_time": r['processing_time'],
            "result_created":  r['result_created'].isoformat() if r['result_created'] else None,
        }

    # Harmonized boxes
    cur.execute(
        "SELECT merged_id, merged_data, status, created FROM merged_boxes WHERE image_id = %s",
        (image_id,),
    )
    merged_boxes = []
    for r in cur.fetchall():
        row_dict = dict(r)
        if row_dict.get('created'):
            row_dict['created'] = row_dict['created'].isoformat()
        merged_boxes.append(row_dict)

    # Consensus
    cur.execute(
        """SELECT consensus_data, processing_time, consensus_created
           FROM consensus WHERE image_id = %s
           ORDER BY consensus_created DESC LIMIT 1""",
        (image_id,),
    )
    consensus_row = cur.fetchone()
    consensus = None
    if consensus_row:
        consensus = {
            "consensus_data":    consensus_row['consensus_data'],
            "processing_time":   consensus_row['processing_time'],
            "consensus_created": consensus_row['consensus_created'].isoformat() if consensus_row['consensus_created'] else None,
        }

    # Content analysis
    cur.execute(
        """SELECT full_analysis, created, analysis_version, processing_time
           FROM content_analysis WHERE image_id = %s""",
        (image_id,),
    )
    content_row      = cur.fetchone()
    content_analysis = dict(content_row) if content_row else None
    if content_analysis and content_analysis.get('created'):
        content_analysis['created'] = content_analysis['created'].isoformat()

    # Postprocessing — also resolve source_bbox for canvas coordinate transforms.
    cur.execute(
        """SELECT p.service, p.merged_box_id, p.data, p.processing_time,
                  mb.merged_data->'merged_bbox' AS source_bbox
           FROM postprocessing p
           LEFT JOIN merged_boxes mb ON mb.merged_id = p.merged_box_id
           WHERE p.image_id = %s AND p.status = 'success'""",
        (image_id,),
    )
    postprocessing = [dict(r) for r in cur.fetchall()]

    # Noun consensus
    cur.execute(
        """SELECT nouns, category_tally, services_present, service_count, processing_time, created_at, updated_at
           FROM noun_consensus WHERE image_id = %s""",
        (image_id,),
    )
    noun_consensus_row = cur.fetchone()
    noun_consensus     = None
    if noun_consensus_row:
        all_nouns       = noun_consensus_row['nouns'] or []
        consensus_nouns = [n for n in all_nouns if n.get('confidence', 0) > 0.5 or n.get('promoted', False)]
        noun_consensus  = {
            "nouns":            consensus_nouns,
            "nouns_all":        all_nouns,
            "category_tally":   noun_consensus_row['category_tally'] or [],
            "services_present": noun_consensus_row['services_present'],
            "service_count":    noun_consensus_row['service_count'],
            "processing_time":  noun_consensus_row['processing_time'],
            "created_at":       noun_consensus_row['created_at'].isoformat() if noun_consensus_row['created_at'] else None,
            "updated_at":       noun_consensus_row['updated_at'].isoformat() if noun_consensus_row['updated_at'] else None,
        }

    # SAM3 segmentation results
    cur.execute(
        """SELECT nouns_queried, data, instance_count, processing_time, created_at, updated_at
           FROM sam3_results WHERE image_id = %s""",
        (image_id,),
    )
    sam3_row    = cur.fetchone()
    sam3_results = None
    if sam3_row:
        sam3_results = {
            "nouns_queried":   sam3_row['nouns_queried'],
            "results":         sam3_row['data'],
            "instance_count":  sam3_row['instance_count'],
            "processing_time": sam3_row['processing_time'],
            "created_at":      sam3_row['created_at'].isoformat() if sam3_row['created_at'] else None,
            "updated_at":      sam3_row['updated_at'].isoformat() if sam3_row['updated_at'] else None,
        }

    # Verb consensus
    cur.execute(
        """SELECT verbs, svo_triples, services_present, service_count, processing_time, created_at, updated_at
           FROM verb_consensus WHERE image_id = %s""",
        (image_id,),
    )
    verb_consensus_row = cur.fetchone()
    verb_consensus     = None
    if verb_consensus_row:
        verb_consensus = {
            "verbs":            verb_consensus_row['verbs'] or [],
            "svo_triples":      verb_consensus_row['svo_triples'] or {},
            "services_present": verb_consensus_row['services_present'],
            "service_count":    verb_consensus_row['service_count'],
            "processing_time":  verb_consensus_row['processing_time'],
            "created_at":       verb_consensus_row['created_at'].isoformat() if verb_consensus_row['created_at'] else None,
            "updated_at":       verb_consensus_row['updated_at'].isoformat() if verb_consensus_row['updated_at'] else None,
        }

    # Caption summary
    cur.execute(
        """SELECT summary_caption, model, services_present, service_count, processing_time, created_at, updated_at
           FROM caption_summary WHERE image_id = %s""",
        (image_id,),
    )
    caption_summary_row = cur.fetchone()
    caption_summary     = None
    if caption_summary_row:
        caption_summary = {
            "summary_caption":  caption_summary_row['summary_caption'],
            "model":            caption_summary_row['model'],
            "services_present": caption_summary_row['services_present'],
            "service_count":    caption_summary_row['service_count'],
            "processing_time":  caption_summary_row['processing_time'],
            "created_at":       caption_summary_row['created_at'].isoformat() if caption_summary_row['created_at'] else None,
            "updated_at":       caption_summary_row['updated_at'].isoformat() if caption_summary_row['updated_at'] else None,
        }

    # Service dispatch — most recent status per (service, cluster_id).
    cur.execute(
        """SELECT DISTINCT ON (service, cluster_id)
                  service, cluster_id, status, failed_reason, dispatched_at
           FROM service_dispatch
           WHERE image_id = %s
           ORDER BY service, cluster_id NULLS LAST, dispatched_at DESC""",
        (image_id,),
    )
    service_dispatch = []
    for r in cur.fetchall():
        row = dict(r)
        if row.get('dispatched_at'):
            row['dispatched_at'] = row['dispatched_at'].isoformat()
        service_dispatch.append(row)

    # Resolve source_bbox for SAM3-dispatched postprocessing rows.
    if sam3_results:
        sam3_data = sam3_results.get('results') or {}
        for row in postprocessing:
            if row.get('source_bbox') is not None:
                continue
            cluster_id = (row.get('data') or {}).get('cluster_id', '')
            if not cluster_id.startswith('sam3:'):
                continue
            parts = cluster_id.split(':')
            if len(parts) == 3:
                _, noun, idx_str = parts
                try:
                    inst = sam3_data.get(noun, {}).get('instances', [])[int(idx_str)]
                    row['source_bbox'] = inst.get('bbox')
                except (IndexError, ValueError, TypeError):
                    pass

    # Aggregate results live in service_results so SDK consumers that iterate
    # that dict (e.g. AnalysisResult._from_status()) can access them via
    # natural attribute access (result.noun_consensus, etc.).
    if noun_consensus is not None:
        service_results["noun_consensus"] = noun_consensus
    if verb_consensus is not None:
        service_results["verb_consensus"] = verb_consensus
    if consensus is not None:
        service_results["consensus"] = consensus
    if caption_summary is not None:
        service_results["caption_summary"] = caption_summary
    if content_analysis is not None:
        service_results["content_analysis"] = content_analysis

    return {
        "service_results":  service_results,
        "merged_boxes":     merged_boxes,
        "postprocessing":   postprocessing,
        "sam3":             sam3_results,
        "service_dispatch": service_dispatch,
    }
