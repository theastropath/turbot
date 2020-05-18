"""Support longer timestamps

Revision ID: a5d26dbfc9a3
Revises: 33940279af35
Create Date: 2020-05-18 14:32:29.147724

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "a5d26dbfc9a3"
down_revision = "33940279af35"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("prices") as b:
        b.alter_column(
            "timestamp", existing_type=sa.String(length=30), type_=sa.String(length=50)
        )


def downgrade():
    with op.batch_alter_table("prices") as b:
        b.alter_column(
            "timestamp", existing_type=sa.String(length=50), type_=sa.String(length=30)
        )
