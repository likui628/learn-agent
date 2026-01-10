"""Small but complete example of using Pydantic AI to build a support agent for a bank.

Run with:

    uv run -m pydantic_ai_examples.bank_support
"""

import os
import sqlite3
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

load_dotenv()


@dataclass
class DatabaseConn:
    """A wrapper over the SQLite connection."""

    sqlite_conn: sqlite3.Connection

    async def customer_name(self, *, id: int) -> str | None:
        res = cur.execute('SELECT name FROM customers WHERE id=?', (id,))
        row = res.fetchone()
        if row:
            return row[0]
        return None

    async def customer_balance(self, *, id: int) -> float:
        res = cur.execute('SELECT balance FROM customers WHERE id=?', (id,))
        row = res.fetchone()
        if row:
            return row[0]
        else:
            raise ValueError('Customer not found')


@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn


class SupportOutput(BaseModel):
    support_advice: str
    """Advice returned to the customer"""
    block_card: bool
    """Whether to block their card or not"""
    risk: int
    """Risk level of query"""


model = OpenRouterModel(
    'mistralai/devstral-2512:free',
    provider=OpenRouterProvider(
        api_key=os.getenv('OPENROUTER_API_KEY'),
        app_url='https://github.com/likui628/pydantic-ai-examples',
        app_title='Pydantic AI examples',
    )
)

support_agent = Agent(
    model=model,
    deps_type=SupportDependencies,
    output_type=SupportOutput,
    instructions=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query. '
        "Reply using the customer's name."
    ),
)


@support_agent.instructions
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


@support_agent.tool
async def customer_balance(ctx: RunContext[SupportDependencies]) -> str:
    """Returns the customer's current account balance."""
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
    )
    return f'${balance:.2f}'


if __name__ == '__main__':
    with sqlite3.connect(':memory:') as con:
        cur = con.cursor()
        cur.execute('CREATE TABLE customers(id, name, balance)')
        cur.execute("""
            INSERT INTO customers VALUES
                (123, 'John', 123.45)
        """)
        cur.execute("""
            INSERT INTO customers VALUES
                (456, 'Jane', 678.90)
        """)
        con.commit()

        def prompt_user_id() -> SupportDependencies:
            while True:
                user_id = input("please enter your user ID: ").strip()
                if not user_id.isdigit():
                    print("Invalid user ID. Please enter numbers only.")
                    continue
                numeric_id = int(user_id)
                cur = con.cursor()
                cur.execute('SELECT 1 FROM customers WHERE id=?',
                            (numeric_id,))
                if cur.fetchone():
                    return SupportDependencies(
                        customer_id=numeric_id, db=DatabaseConn(
                            sqlite_conn=con)
                    )
                print("Customer not found. Please enter a valid user ID.")

        deps = prompt_user_id()
        print("Asking me anything about your bank account, enter 'exit', 'quit', or 'q' to leave.\n")

        while True:
            user_input = input("Your question: ").strip()
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            try:
                result = support_agent.run_sync(user_input, deps=deps)
            except Exception as e:
                print("Something went wrong, please try again:")
                continue
            print(result.output)
            print("\n")
        print("Goodbye!")
