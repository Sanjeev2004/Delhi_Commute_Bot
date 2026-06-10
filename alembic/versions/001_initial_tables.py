"""initial tables

Revision ID: 001
Revises:
Create Date: 2026-06-10

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- query_logs ---
    op.create_table(
        "query_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("user_phone_hash", sa.String(length=128), nullable=False),
        sa.Column("raw_query", sa.Text(), nullable=False),
        sa.Column("intent", sa.String(length=64), nullable=False),
        sa.Column("source_location", sa.String(length=256), nullable=True),
        sa.Column("destination_location", sa.String(length=256), nullable=True),
        sa.Column("response_text", sa.Text(), nullable=False),
        sa.Column("response_time_ms", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_query_logs_user_phone_hash"),
        "query_logs",
        ["user_phone_hash"],
        unique=False,
    )
    op.create_index(
        op.f("ix_query_logs_intent"),
        "query_logs",
        ["intent"],
        unique=False,
    )

    # --- popular_routes ---
    op.create_table(
        "popular_routes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source", sa.String(length=256), nullable=False),
        sa.Column("destination", sa.String(length=256), nullable=False),
        sa.Column(
            "query_count", sa.Integer(), nullable=False, server_default=sa.text("1")
        ),
        sa.Column(
            "last_queried",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_popular_routes_source"),
        "popular_routes",
        ["source"],
        unique=False,
    )
    op.create_index(
        op.f("ix_popular_routes_destination"),
        "popular_routes",
        ["destination"],
        unique=False,
    )

    # --- user_feedback ---
    op.create_table(
        "user_feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("query_log_id", sa.Integer(), nullable=False),
        sa.Column("user_phone_hash", sa.String(length=128), nullable=True),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "is_incorrect",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["query_log_id"],
            ["query_logs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_user_feedback_query_log_id"),
        "user_feedback",
        ["query_log_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_user_feedback_user_phone_hash"),
        "user_feedback",
        ["user_phone_hash"],
        unique=False,
    )

    # --- conversation_sessions ---
    op.create_table(
        "conversation_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("user_phone_hash", sa.String(length=128), nullable=False),
        sa.Column("context", sa.Text(), nullable=True),
        sa.Column("last_intent", sa.String(length=64), nullable=True),
        sa.Column("last_source", sa.String(length=256), nullable=True),
        sa.Column("last_destination", sa.String(length=256), nullable=True),
        sa.Column(
            "message_count", sa.Integer(), server_default=sa.text("0"), nullable=True
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_conversation_sessions_session_id"),
        "conversation_sessions",
        ["session_id"],
        unique=True,
    )
    op.create_index(
        op.f("ix_conversation_sessions_user_phone_hash"),
        "conversation_sessions",
        ["user_phone_hash"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_table("conversation_sessions")
    op.drop_table("user_feedback")
    op.drop_table("popular_routes")
    op.drop_table("query_logs")
