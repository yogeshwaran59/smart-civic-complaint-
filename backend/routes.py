import os
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, Response
from werkzeug.utils import secure_filename
from models import db, User, Complaint, StatusLog
from ai_processor import classify_complaint_text, haversine_distance, get_image_similarity, analyze_and_describe_image

routes_bp = Blueprint('routes', __name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function for smart ward routing
def get_ward_by_location(lat, lng):
    # Center of Bangalore-like coords (12.97, 77.59)
    # Simple bounding boxes for ward routing
    if lat > 12.97:
        return 'ward_1' if lng < 77.59 else 'ward_2'
    else:
        return 'ward_3'

@routes_bp.route('/api/complaints', methods=['POST'])
def create_complaint():
    """
    Creates a new complaint. Runs AI duplicate check and text classification.
    Form data:
    - description (text)
    - latitude (float)
    - longitude (float)
    - contact (text)
    - image (file, optional)
    """
    try:
        description = request.form.get('description', '')
        latitude_str = request.form.get('latitude')
        longitude_str = request.form.get('longitude')
        contact = request.form.get('contact', '')

        if not description or not latitude_str or not longitude_str:
            return jsonify({'error': 'Description, latitude, and longitude are required.'}), 400

        try:
            latitude = float(latitude_str)
            longitude = float(longitude_str)
        except ValueError:
            return jsonify({'error': 'Invalid latitude or longitude values.'}), 400

        # Create unique ID for complaint
        complaint_id = f"COMP-{uuid.uuid4().hex[:6].upper()}"

        # Handle image upload
        image_file = request.files.get('image')
        image_path = None
        if image_file and image_file.filename:
            filename = f"{complaint_id}_{secure_filename(image_file.filename)}"
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            image_file.save(save_path)
            image_path = f"/uploads/{filename}"

        # Smart Ward Routing
        ward = get_ward_by_location(latitude, longitude)

        # AI Classifier: Auto category & priority
        category, priority = classify_complaint_text(description)

        # AI Duplicate Check: Find nearby complaints (50m radius)
        # and compare images if both exist
        duplicate_flag = False
        duplicate_of = None
        existing_complaints = Complaint.query.filter(
            Complaint.status.in_(['Submitted', 'Assigned', 'In Progress'])
        ).all()

        for ext in existing_complaints:
            # Geographic check
            dist = haversine_distance(latitude, longitude, ext.latitude, ext.longitude)
            if dist <= 50.0:  # 50 meters
                # If images are present, check image similarity
                if image_path and ext.image_path:
                    abs_path_new = os.path.join(os.path.dirname(__file__), image_path.lstrip('/'))
                    abs_path_ext = os.path.join(os.path.dirname(__file__), ext.image_path.lstrip('/'))
                    similarity = get_image_similarity(abs_path_new, abs_path_ext)
                    if similarity > 0.82:
                        duplicate_flag = True
                        duplicate_of = ext.complaint_id
                        break
                else:
                    # If no images, check if category matches and descriptions are similar
                    if category == ext.category:
                        duplicate_flag = True
                        duplicate_of = ext.complaint_id
                        break

        # Run image analysis
        image_analysis = None
        if image_path:
            abs_path_new = os.path.join(os.path.dirname(__file__), image_path.lstrip('/'))
            image_analysis = analyze_and_describe_image(abs_path_new)

        # Save to database
        new_complaint = Complaint(
            complaint_id=complaint_id,
            description=description,
            image_path=image_path,
            latitude=latitude,
            longitude=longitude,
            category=category,
            priority=priority,
            status='Submitted',
            created_at=datetime.utcnow(),
            escalation_flag=duplicate_flag,  # We can flag it, or keep track of duplicate
            ward=ward,
            image_analysis=image_analysis
        )
        
        # Add to session
        db.session.add(new_complaint)
        db.session.flush() # Populate models to write to log

        # Initial Status Log
        log_status = "Submitted"
        if duplicate_flag:
            log_status = f"Submitted (Duplicate of {duplicate_of})"
            
        log = StatusLog(
            complaint_id=complaint_id,
            status=log_status,
            timestamp=datetime.utcnow()
        )
        db.session.add(log)
        db.session.commit()

        # Print mock SMS log
        print(f"============================================================")
        print(f"[NEW] COMPLAINT RECEIVED: {complaint_id}")
        print(f"Location: {latitude}, {longitude} -> Assigned to Ward: {ward}")
        print(f"AI Categorization: Category='{category}', Priority='{priority}'")
        if duplicate_flag:
            print(f"WARNING: AI DUPLICATE CHECK: Flagged as DUPLICATE of {duplicate_of}")
        print(f"MOCK SMS SENT TO CITIZEN ({contact}): "
              f"'Thank you for reporting! Your complaint ID is {complaint_id}. Status: {log_status}. Ward: {ward}.'")
        print(f"============================================================")

        response_data = new_complaint.to_dict()
        response_data['is_duplicate'] = duplicate_flag
        response_data['duplicate_of'] = duplicate_of

        return jsonify(response_data), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@routes_bp.route('/api/complaints', methods=['GET'])
def get_complaints():
    """
    Get list of complaints. Supports filtering by ward, status, priority, category.
    """
    try:
        query = Complaint.query
        
        # Apply filters
        ward = request.args.get('ward')
        status = request.args.get('status')
        priority = request.args.get('priority')
        category = request.args.get('category')
        redirected_to_journalist = request.args.get('redirected_to_journalist')
        created_after = request.args.get('created_after')

        if ward:
            query = query.filter_by(ward=ward)
        if status:
            query = query.filter_by(status=status)
        if priority:
            query = query.filter_by(priority=priority)
        if category:
            query = query.filter_by(category=category)
        if redirected_to_journalist is not None:
            val = redirected_to_journalist.lower() in ['true', '1']
            query = query.filter_by(redirected_to_journalist=val)
        if created_after:
            try:
                date_obj = datetime.fromisoformat(created_after)
                query = query.filter(Complaint.created_at >= date_obj)
            except ValueError:
                pass

        # Order by creation date descending
        complaints = query.order_by(Complaint.created_at.desc()).all()
        return jsonify([c.to_dict() for c in complaints]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@routes_bp.route('/api/complaints/<id>', methods=['GET'])
def get_complaint_by_id(id):
    """
    Get a single complaint and its timeline history.
    """
    try:
        complaint = Complaint.query.get(id)
        if not complaint:
            return jsonify({'error': 'Complaint not found'}), 404
            
        logs = StatusLog.query.filter_by(complaint_id=id).order_by(StatusLog.timestamp.asc()).all()
        
        data = complaint.to_dict()
        data['history'] = [l.to_dict() for l in logs]
        return jsonify(data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@routes_bp.route('/api/complaints/<id>', methods=['PUT'])
def update_complaint(id):
    """
    Updates status, worker assignment.
    Body JSON:
    - status ('Assigned' | 'In Progress' | 'Resolved' | 'Closed')
    - assigned_to (int, optional)
    - resolution_image (handled via multi-part or base64, but since it could be PUT form we support both JSON and form data)
    """
    try:
        complaint = Complaint.query.get(id)
        if not complaint:
            return jsonify({'error': 'Complaint not found'}), 404

        # Read JSON or Form data
        if request.is_json:
            data = request.json
            status = data.get('status')
            assigned_to = data.get('assigned_to')
        else:
            data = request.form
            status = data.get('status')
            assigned_to = data.get('assigned_to')

        if not status:
            return jsonify({'error': 'Status is required'}), 400

        # Handle worker assignment
        if assigned_to is not None:
            if assigned_to == "":
                complaint.assigned_to = None
            else:
                try:
                    worker_id = int(assigned_to)
                    worker = User.query.get(worker_id)
                    if worker:
                        complaint.assigned_to = worker_id
                        # If assigning, set status to 'Assigned' if it was 'Submitted'
                        if status == 'Submitted' or not status:
                            status = 'Assigned'
                except ValueError:
                    pass

        # Handle priority update
        priority = data.get('priority')
        if priority:
            complaint.priority = priority

        # Manage dates
        if status in ['Assigned', 'In Progress'] and not complaint.opened_at:
            complaint.opened_at = datetime.utcnow()

        # Handle resolution image (if file uploaded during PUT)
        resolution_file = request.files.get('resolution_image') if not request.is_json else None
        if resolution_file and resolution_file.filename:
            filename = f"RESOLVED_{id}_{secure_filename(resolution_file.filename)}"
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            resolution_file.save(save_path)
            # Update complaint description or store in database. For simplicity, we can append to image_path 
            # or log it. Let's update image_path or log it. We can set description or keep it.
            # Let's save the resolution photo in status logs or log text.
            # To keep things simple, we prepend the resolution image path to the description or log it.
            # Or we can update the image_path to the resolved image to show it on UI!
            complaint.image_path = f"/uploads/{filename}"

        # Update status
        old_status = complaint.status
        complaint.status = status

        # Read optional progress/resolution notes
        notes = data.get('notes') or data.get('resolution_notes')

        # Log transition
        log = StatusLog(
            complaint_id=id,
            status=status,
            notes=notes,
            timestamp=datetime.utcnow()
        )
        db.session.add(log)
        db.session.commit()

        # Print mock SMS logs
        print(f"============================================================")
        print(f"STATUS UPDATE: {id}")
        print(f"   State change: '{old_status}' -> '{status}'")
        if complaint.assigned_to:
            worker_name = complaint.assigned_worker.name if complaint.assigned_worker else "Worker"
            print(f"   Assigned Worker ID: {complaint.assigned_to} ({worker_name})")
        print(f"MOCK SMS SENT: 'Complaint {id} status updated to {status}.'")
        print(f"============================================================")

        return jsonify(complaint.to_dict()), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@routes_bp.route('/api/users', methods=['GET'])
def get_users():
    """
    Get users (workers or authorities). Filtering by role.
    """
    try:
        role = request.args.get('role')
        ward = request.args.get('ward')
        query = User.query
        
        if role:
            query = query.filter_by(role=role)
        if ward:
            query = query.filter_by(ward=ward)
            
        users = query.all()
        return jsonify([u.to_dict() for u in users]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@routes_bp.route('/api/analytics', methods=['GET'])
def get_analytics():
    """
    Aggregate metrics for Chart.js.
    """
    try:
        query = Complaint.query
        created_after = request.args.get('created_after')
        if created_after:
            try:
                date_obj = datetime.fromisoformat(created_after)
                query = query.filter(Complaint.created_at >= date_obj)
            except ValueError:
                pass
        complaints = query.all()
        
        total = len(complaints)
        resolved = sum(1 for c in complaints if c.status == 'Resolved')
        pending = sum(1 for c in complaints if c.status in ['Submitted', 'Assigned', 'In Progress'])
        escalated = sum(1 for c in complaints if c.escalation_flag)

        # Category Counts
        categories = {'pothole': 0, 'garbage': 0, 'drainage': 0, 'street_light': 0, 'other': 0}
        # Status Counts
        statuses = {'Submitted': 0, 'Assigned': 0, 'In Progress': 0, 'Resolved': 0, 'Closed': 0}
        # Ward Counts
        wards = {'ward_1': 0, 'ward_2': 0, 'ward_3': 0}

        for c in complaints:
            if c.category in categories:
                categories[c.category] += 1
            if c.status in statuses:
                statuses[c.status] += 1
            if c.ward in wards:
                wards[c.ward] += 1

        return jsonify({
            'total': total,
            'resolved': resolved,
            'pending': pending,
            'escalated': escalated,
            'categories': categories,
            'statuses': statuses,
            'wards': wards
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# REAL EXOTEL IVR WEBHOOKS
import requests

@routes_bp.route('/api/exotel/webhook', methods=['POST', 'GET'])
def exotel_webhook():
    """
    Exotel Passthru Webhook.
    Handles the final step of an Exotel Applet.
    Expects POST data: From, RecordingUrl, digits (custom variable).
    """
    try:
        data = request.form if request.method == 'POST' else request.args
        caller_phone = data.get('From', '')
        recording_url = data.get('RecordingUrl', '')
        digits = data.get('digits', '5') # Assume standard Exotel custom field or passthru param

        # Map digits to category
        category_map = {
            '1': 'pothole',
            '2': 'garbage',
            '3': 'drainage',
            '4': 'street_light',
            '5': 'other'
        }
        category = category_map.get(digits, 'other')

        # Generate default GPS coordinates (e.g. Center of Bangalore-like environment)
        import random
        latitude = 12.971598 + random.uniform(-0.02, 0.02)
        longitude = 77.594562 + random.uniform(-0.02, 0.02)
        
        complaint_id = f"COMP-IVR-{uuid.uuid4().hex[:4].upper()}"
        ward = get_ward_by_location(latitude, longitude)
        
        priority = 'High' if category in ['drainage', 'pothole'] else 'Medium'
        desc = f"Reported via Exotel IVR Hotline. Voice Message: {recording_url}"

        # Save complaint
        new_complaint = Complaint(
            complaint_id=complaint_id,
            description=desc,
            image_path=None,
            latitude=latitude,
            longitude=longitude,
            category=category,
            priority=priority,
            status='Submitted',
            created_at=datetime.utcnow(),
            escalation_flag=False,
            ward=ward
        )
        db.session.add(new_complaint)
        db.session.flush()

        log = StatusLog(
            complaint_id=complaint_id,
            status="Submitted (Exotel IVR)",
            timestamp=datetime.utcnow()
        )
        db.session.add(log)
        db.session.commit()

        # Send Real SMS via Exotel
        exotel_sid = os.environ.get('EXOTEL_SID')
        exotel_key = os.environ.get('EXOTEL_API_KEY')
        exotel_token = os.environ.get('EXOTEL_API_TOKEN')
        exotel_subdomain = os.environ.get('EXOTEL_SUBDOMAIN', 'api.exotel.com')
        exotel_caller_id = os.environ.get('EXOTEL_CALLER_ID')
        
        if exotel_sid and exotel_key and exotel_token and exotel_caller_id and caller_phone:
            try:
                sms_url = f"https://{exotel_key}:{exotel_token}@{exotel_subdomain}/v1/Accounts/{exotel_sid}/Sms/send.json"
                sms_data = {
                    "From": exotel_caller_id,
                    "To": caller_phone,
                    "Body": f"SmartCivic: Thank you for your call. Your complaint tracking ID is {complaint_id}. Ward: {ward}"
                }
                response = requests.post(sms_url, data=sms_data)
                
                if response.status_code == 200:
                    print(f"[IVR] Exotel SMS sent to {caller_phone}.")
                else:
                    print(f"[IVR] Failed to send Exotel SMS: {response.text}")
            except Exception as sms_error:
                print(f"[IVR] Failed to send Exotel SMS: {str(sms_error)}")
        else:
            print(f"[IVR] Exotel credentials not fully configured. Cannot send real SMS to {caller_phone}.")

        print(f"============================================================")
        print(f"EXOTEL IVR CALL & RECORDING SUCCESSFUL")
        print(f"   Created complaint: {complaint_id}")
        print(f"   Category: {category}")
        print(f"   Recording URL: {recording_url}")
        print(f"   Location: {latitude}, {longitude} (Ward: {ward})")
        print(f"============================================================")

        return jsonify({"status": "success", "complaint_id": complaint_id}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


# AUTHENTICATION ENDPOINTS
@routes_bp.route('/api/auth/signup', methods=['POST'])
def signup():
    """
    Register a new user account.
    """
    try:
        data = request.json
        name = data.get('name')
        gmail = data.get('gmail')
        password = data.get('password')
        role = data.get('role')  # 'citizen' | 'worker' | 'authority'
        contact = data.get('contact')
        ward = data.get('ward', 'general')

        if not name or not gmail or not password or not role or not contact:
            return jsonify({'error': 'Name, Gmail, Password, Role, and Contact are required.'}), 400

        # Check existing user
        existing_user = User.query.filter_by(gmail=gmail).first()
        if existing_user:
            return jsonify({'error': 'A user with this Gmail address already exists.'}), 409

        new_user = User(
            name=name,
            gmail=gmail,
            password=password,
            role=role,
            contact=contact,
            ward=ward
        )
        db.session.add(new_user)
        db.session.commit()

        print(f"============================================================")
        print(f"[AUTH] New user registered: {name} ({role}) - Gmail: {gmail}")
        print(f"============================================================")

        return jsonify(new_user.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@routes_bp.route('/api/auth/login', methods=['POST'])
def login():
    """
    Authenticate a user and return their details.
    """
    try:
        data = request.json
        gmail = data.get('gmail')
        password = data.get('password')

        if not gmail or not password:
            return jsonify({'error': 'Gmail and Password are required.'}), 400

        user = User.query.filter_by(gmail=gmail).first()
        if not user or user.password != password:
            return jsonify({'error': 'Invalid Gmail address or password.'}), 401

        print(f"============================================================")
        print(f"[AUTH] User logged in: {user.name} ({user.role})")
        print(f"============================================================")

        return jsonify(user.to_dict()), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# JOURNALIST REPORTS ENDPOINTS
from models import JournalistReport

@routes_bp.route('/api/journalist/reports/generate', methods=['POST'])
def generate_journalist_report():
    try:
        data = request.json
        complaint_id = data.get('complaint_id')
        if not complaint_id:
            return jsonify({'error': 'Complaint ID is required'}), 400
            
        complaint = Complaint.query.get(complaint_id)
        if not complaint:
            return jsonify({'error': 'Complaint not found'}), 404
            
        # Simulate AI Agent news report generation
        category_title = complaint.category.replace('_', ' ').upper()
        
        # Build news title
        title = f"INVESTIGATIVE REPORT: Unaddressed {category_title} Neglect in Smart City Ward"
        
        # Build detailed news content
        content = (
            f"--- CITY JOURNAL WATCHDOG ---\n\n"
            f"MUNICIPAL TIMEOUT ACTION: Complaint {complaint_id} has breached the standard 5-minute authority response threshold. "
            f"The civic issue, categorized as a '{complaint.category}' threat, remains unassigned and unresolved.\n\n"
            f"GEOGRAPHIC LOCATION:\n"
            f"The issue is pinned at Latitude: {complaint.latitude:.6f}, Longitude: {complaint.longitude:.6f}.\n\n"
            f"CITIZEN TESTIMONY & ANALYSIS:\n"
            f"\" {complaint.description} \"\n\n"
        )
        
        if complaint.image_analysis:
            content += (
                f"COMPUTER VISION AUDIT REPORT:\n"
                f"Advanced image telemetry scans indicate critical structural features: {complaint.image_analysis}\n\n"
            )
            
        content += (
            f"PRESS WATCHDOG VERDICT:\n"
            f"Due to complete lack of administrative action within the designated window, this case is flagged for general public press release. "
            f"SmartCivic AI watchdog agent recommends immediate worker allocation before severe safety incidents occur."
        )
        
        return jsonify({
            'complaint_id': complaint_id,
            'title': title,
            'content': content
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@routes_bp.route('/api/journalist/reports', methods=['POST'])
def save_journalist_report():
    try:
        data = request.json
        complaint_id = data.get('complaint_id')
        title = data.get('title')
        content = data.get('content')
        published = data.get('published', False)

        if not complaint_id or not title or not content:
            return jsonify({'error': 'complaint_id, title, and content are required'}), 400

        new_report = JournalistReport(
            complaint_id=complaint_id,
            title=title,
            content=content,
            published=published
        )
        db.session.add(new_report)
        db.session.commit()

        print(f"============================================================")
        print(f"[PRESS RELEASE] Journalist saved report for {complaint_id}")
        if published:
            print(f"   STATUS: PUBLISHED TO FEED!")
        print(f"============================================================")

        return jsonify(new_report.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@routes_bp.route('/api/journalist/reports', methods=['GET'])
def get_journalist_reports():
    try:
        reports = JournalistReport.query.order_by(JournalistReport.created_at.desc()).all()
        return jsonify([r.to_dict() for r in reports]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@routes_bp.route('/api/journalist/reports/<int:id>', methods=['PUT'])
def update_journalist_report(id):
    try:
        data = request.json
        report = JournalistReport.query.get(id)
        if not report:
            return jsonify({'error': 'Report not found'}), 404

        if 'title' in data:
            report.title = data['title']
        if 'content' in data:
            report.content = data['content']
        if 'published' in data:
            report.published = data['published']

        db.session.commit()

        if report.published:
            print(f"============================================================")
            print(f"[PRESS RELEASE] Article ID {id} has been PUBLISHED!")
            print(f"   Title: '{report.title}'")
            print(f"============================================================")

        return jsonify(report.to_dict()), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
