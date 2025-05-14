from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from models import db, ForumPost, ForumComment, ForumAttachment
from werkzeug.utils import secure_filename
import os

forum_bp = Blueprint('forum', __name__)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'py', 'zip'}

@forum_bp.route("/forum")
def forum_feed():
    posts = ForumPost.query.options(
        db.joinedload(ForumPost.attachments)
    ).order_by(ForumPost.created_at.desc()).all()
    
    # Debug: Print number of posts and their attachments
    print(f"Total posts: {len(posts)}")
    for post in posts:
        print(f"Post ID: {post.id} - Attachments: {len(post.attachments)}")
    
    return render_template("forum_feed.html", posts=posts)

@forum_bp.route("/forum/post/<int:post_id>", methods=["GET", "POST"])
def forum_post(post_id):
    # Load post with attachments and comments
    post = ForumPost.query.options(
        db.joinedload(ForumPost.attachments),
        db.joinedload(ForumPost.comments).joinedload(ForumComment.user)
    ).get_or_404(post_id)
    
    # Debug: Print post and attachment info
    print(f"Viewing post ID: {post.id}")
    print(f"Attachments count: {len(post.attachments)}")
    for attachment in post.attachments:
        print(f" - Attachment: {attachment.filename} (ID: {attachment.id})")

    if request.method == "POST":
        content = request.form["content"]
        files = request.files.getlist("files")
        
        # Debug: Print received files
        print(f"Received {len(files)} files in request")
        for file in files:
            print(f" - File: {file.filename} (size: {len(file.read())} bytes)")
            file.seek(0)  # Reset file pointer after reading

        comment = ForumComment(
            post_id=post.id,
            user_id=current_user.id,
            content=content
        )
        db.session.add(comment)
        
        # Handle file uploads
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filedata = file.read()
                
                attachment = ForumAttachment(
                    filename=filename,
                    filedata=filedata,
                    post_id=post.id
                )
                db.session.add(attachment)
                print(f"Saved attachment: {filename}")
            else:
                print(f"Skipped invalid file: {file.filename}")
        
        db.session.commit()
        flash("Comment and attachments saved successfully!", "success")
        return redirect(url_for("forum.forum_post", post_id=post_id))
    
    return render_template("forum_post.html", post=post)

@forum_bp.route("/forum/new", methods=["GET", "POST"])
@login_required
def new_post():
    if request.method == "POST":
        title = request.form["title"]
        content = request.form["content"]
        files = request.files.getlist("files")
        
        # Debug: Print received files
        print(f"Creating new post with {len(files)} attachments")
        for file in files:
            print(f" - File: {file.filename}")

        post = ForumPost(
            title=title,
            content=content,
            user_id=current_user.id
        )
        db.session.add(post)
        db.session.flush()  # Generate ID without committing
        
        # Save attachments
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filedata = file.read()
                
                attachment = ForumAttachment(
                    filename=filename,
                    filedata=filedata,
                    post_id=post.id
                )
                db.session.add(attachment)
                print(f"Saved attachment: {filename}")
            else:
                print(f"Skipped invalid file: {file.filename}")
        
        db.session.commit()
        flash("Post created successfully!", "success")
        return redirect(url_for("forum.forum_feed"))
    
    return render_template("new_post.html")