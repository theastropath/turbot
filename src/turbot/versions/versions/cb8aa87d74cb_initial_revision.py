"""Initial revision

Revision ID: cb8aa87d74cb
Revises:
Create Date: 2020-05-15 14:27:38.381917

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "cb8aa87d74cb"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "art",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("author", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=50), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "bugs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("author", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=50), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "fish",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("author", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=50), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "fossils",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("author", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=50), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "prices",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("author", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("price", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.String(length=30), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "songs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("author", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=50), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "users",
        sa.Column("author", sa.Integer(), nullable=False),
        sa.Column("hemisphere", sa.String(length=50), nullable=True),
        sa.Column("timezone", sa.String(length=50), nullable=True),
        sa.Column("island", sa.String(length=50), nullable=True),
        sa.Column("friend", sa.String(length=20), nullable=True),
        sa.Column("fruit", sa.String(length=10), nullable=True),
        sa.Column("nickname", sa.String(length=50), nullable=True),
        sa.Column("creator", sa.String(length=20), nullable=True),
        sa.PrimaryKeyConstraint("author"),
    )


def downgrade():
    op.drop_table("users")
    op.drop_table("songs")
    op.drop_table("prices")
    op.drop_table("fossils")
    op.drop_table("fish")
    op.drop_table("bugs")
    op.drop_table("art")
