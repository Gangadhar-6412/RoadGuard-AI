from database import event
def notify(report_id, work_id, contractor):
    event('Notification', f'Simulated assignment notification queued for {contractor} ({work_id}). Configure SMTP separately for email.', report_id)
