"""
Database utilities for user management and chat history
"""
import os
from contextlib import contextmanager

import psycopg2
from config import MODEL
from openai import OpenAI
from psycopg2.extras import Json, RealDictCursor


def generate_conversation_title(
    first_message: str,
    client: OpenAI = None,
    model: str = MODEL
) -> str:
    """
    Use LLM to generate a concise conversation title from the first user message.
    Returns a short title (max 30 chars) summarizing the conversation topic.

    Args:
        first_message: The user's first message in the conversation
        client: OpenAI client instance (optional, creates new one if not provided)
        model: Model to use for title generation
    """
    if not first_message or not first_message.strip():
        return "New conversation"

    try:
        if client is None:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a very short title (max 6 words, under 30 characters) for this conversation. "
                        "The title should capture the main topic or intent. "
                        "Do NOT use quotes or punctuation. Just output the title directly. "
                    )
                },
                {"role": "user", "content": first_message}
            ],
            max_completion_tokens=30,
            reasoning_effort="none",
            temperature=0.3,
        )
        title = response.choices[0].message.content.strip()
        # Ensure title isn't too long
        if len(title) > 50:
            title = title[:47] + "..."
        return title or "New conversation"
    except Exception:
        # Fallback to truncation if LLM fails
        return first_message[:50] + "..." if len(first_message) > 50 else first_message


def get_db_config():
    """Get database configuration from environment variables"""
    return {
        "host": os.getenv("POSTGRES_HOST"),
        "port": int(os.getenv("POSTGRES_PORT")),
        "database": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    }


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = psycopg2.connect(**get_db_config())
        yield conn
    finally:
        if conn:
            conn.close()


def get_or_create_user(oauth_provider: str, email: str, display_name: str = None, avatar_url: str = None) -> dict:
    """
    Get existing user or create new one based on OAuth provider and email.
    Uses email as oauth_user_id since it should be unique.
    Returns user dict with id and other fields.
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Try to find existing user
            cur.execute(
                """
                SELECT id, oauth_provider, oauth_user_id, email, display_name, avatar_url, created_at, last_login_at
                FROM users
                WHERE oauth_provider = %s AND oauth_user_id = %s
                """,
                (oauth_provider, email)
            )
            user = cur.fetchone()

            if user:
                # Update last login time
                cur.execute(
                    "UPDATE users SET last_login_at = NOW() WHERE id = %s",
                    (user["id"],)
                )
                conn.commit()
                return dict(user)

            # Create new user
            cur.execute(
                """
                INSERT INTO users (oauth_provider, oauth_user_id, email, display_name, avatar_url, last_login_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                RETURNING id, oauth_provider, oauth_user_id, email, display_name, avatar_url, created_at, last_login_at
                """,
                (oauth_provider, email, email, display_name, avatar_url)
            )
            new_user = cur.fetchone()
            conn.commit()
            return dict(new_user)


def save_chat_message(user_id: str, role: str, content: str, conversation_id: str) -> int:
    """Save a chat message to the database"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages (user_id, role, content, conversation_id)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (user_id, role, content, conversation_id)
            )
            message_id = cur.fetchone()[0]
            conn.commit()
            return message_id


def get_chat_history(user_id: str, conversation_id: str = None, limit: int = 50) -> list:
    """Get chat history for a user"""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if conversation_id:
                cur.execute(
                    """
                    SELECT role, content, created_at
                    FROM chat_messages
                    WHERE user_id = %s AND conversation_id = %s
                    ORDER BY created_at ASC
                    LIMIT %s
                    """,
                    (user_id, conversation_id, limit)
                )
            else:
                cur.execute(
                    """
                    SELECT role, content, created_at
                    FROM chat_messages
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (user_id, limit)
                )
            return [dict(row) for row in cur.fetchall()]


def record_token_usage(
    user_id: str,
    tokens_in: int,
    tokens_out: int,
    model: str = None,
    conversation_id: str = None
):
    """Record token usage for a user request"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO token_usage (user_id, conversation_id, tokens_in, tokens_out, model)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (user_id, conversation_id, tokens_in, tokens_out, model)
            )
            conn.commit()


def count_recent_messages(user_id: str, hours: int = 24) -> int:
    """Number of messages the user had answered in the last `hours` hours."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM token_usage
                WHERE user_id = %s AND created_at > NOW() - make_interval(hours => %s)
                """,
                (user_id, hours)
            )
            return cur.fetchone()[0]


def get_user_token_stats(user_id: str) -> dict:
    """Get total token usage stats for a user"""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    COALESCE(SUM(tokens_in), 0) as total_tokens_in,
                    COALESCE(SUM(tokens_out), 0) as total_tokens_out,
                    COUNT(*) as total_requests
                FROM token_usage
                WHERE user_id = %s
                """,
                (user_id,)
            )
            return dict(cur.fetchone())


def save_conversation(
    user_id: str,
    conversation_id: str,
    messages: list,
    title: str = None,
    openai_client: OpenAI = None
) -> None:
    """Save or update a conversation (client.messages) to database"""
    if not title and messages:
        # Use LLM to generate a concise title from the first user message
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                title = generate_conversation_title(content, client=openai_client)
                break
        title = title or "New conversation"

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (user_id, conversation_id, title, messages, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (conversation_id)
                DO UPDATE SET messages = %s, title = COALESCE(%s, conversations.title), updated_at = NOW()
                """,
                (user_id, conversation_id, title, Json(messages), Json(messages), title)
            )
            conn.commit()


def get_conversation(conversation_id: str) -> dict | None:
    """Get a conversation by its ID"""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT conversation_id, title, messages, created_at, updated_at
                FROM conversations
                WHERE conversation_id = %s
                """,
                (conversation_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None


def get_user_conversations(user_id: str, limit: int = 20) -> list:
    """Get list of conversations for a user, sorted by most recent"""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT conversation_id, title, created_at, updated_at
                FROM conversations
                WHERE user_id = %s
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (user_id, limit)
            )
            return [dict(row) for row in cur.fetchall()]


def delete_conversation(conversation_id: str) -> None:
    """Delete a conversation"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM conversations WHERE conversation_id = %s",
                (conversation_id,)
            )
            conn.commit()


def delete_user_conversations(user_id: str) -> int:
    """Delete all conversations for a user. Returns number of deleted rows."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM conversations WHERE user_id = %s",
                (user_id,)
            )
            deleted_count = cur.rowcount
            conn.commit()
            return deleted_count
