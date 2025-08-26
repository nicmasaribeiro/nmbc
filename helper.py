from models import UserNotebook, NotebookShare, Users, NotebookInvite
from sqlalchemy import or_
from datetime import datetime, timedelta
from hashlib import sha256
import secrets

def user_role_for_notebook(user_id: int, nb: UserNotebook) -> str | None:
    if nb.user_id == user_id:
        return "owner"
    share = NotebookShare.query.filter_by(notebook_id=nb.id, user_id=user_id).first()
    return share.role if share else None

def assert_access(notebook_id: int, user_id: int, min_role: str = "viewer"):
    nb = UserNotebook.query.get(notebook_id)
    if not nb:
        return None, ("not_found", "Notebook not found")
    role = user_role_for_notebook(user_id, nb)
    ranks = {"viewer": 1, "editor": 2, "owner": 3}
    need = ranks[min_role]
    have = ranks["owner"] if role is None and nb.user_id == user_id else ranks.get(role or "", 0)
    if have < need:
        return None, ("forbidden", "Insufficient permissions")
    return nb, None

def find_user_by_username_or_email(identifier: str) -> Users | None:
    q = Users.query.filter(or_(Users.username==identifier, Users.email==identifier))
    return q.first()

def create_invite(nb_id: int, created_by: int, role: str = "viewer", ttl_minutes: int = 120):
    raw = secrets.token_urlsafe(24)
    token_hash = sha256(raw.encode()).hexdigest()
    inv = NotebookInvite(
        notebook_id=nb_id,
        token_hash=token_hash,
        role=role,
        created_by=created_by,
        expires_at=datetime.utcnow() + timedelta(minutes=ttl_minutes)
    )
    db.session.add(inv)
    db.session.commit()
    return raw  # return raw token (store only hash in DB)

def accept_invite(raw_token: str, user_id: int):
    token_hash = sha256(raw_token.encode()).hexdigest()
    inv = NotebookInvite.query.filter_by(token_hash=token_hash, used=False).first()
    if not inv:
        return None, "Invalid invite"
    if inv.expires_at and inv.expires_at < datetime.utcnow():
        return None, "Invite expired"
    # upsert share
    share = NotebookShare.query.filter_by(notebook_id=inv.notebook_id, user_id=user_id).first()
    if not share:
        share = NotebookShare(notebook_id=inv.notebook_id, user_id=user_id, role=inv.role)
        db.session.add(share)
    else:
        share.role = inv.role
    inv.used = True
    db.session.commit()
    return share, None
