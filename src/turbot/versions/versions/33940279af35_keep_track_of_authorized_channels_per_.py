"""Keep track of authorized channels per server

Revision ID: 33940279af35
Revises: 9b107d322c46
Create Date: 2020-05-17 18:42:33.659939

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "33940279af35"
down_revision = "9b107d322c46"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "authorized_channels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("guild", sa.BIGINT(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade():
    op.drop_table("authorized_channels")
